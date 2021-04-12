import gc
import os
from glob import glob
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import metrics

def plot(X, y, figname=None):
    if len(np.unique(y)) <= 10:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = plt.get_cmap('tab20')

    fig, ax = plt.subplots(figsize=(20, 20))
    
    for cl in np.unique(y):
        ax.scatter(X[y==cl,0], X[y==cl,1], c=[cmap(cl)], label=cl, s=20)
        ax.axis('off')
                        
    ax.legend()

    if figname is not None:
        fig.savefig(figname)

    plt.close('all')
    del fig
    del ax

def cluster(C, X):
    t0 = perf_counter()
    y_pred = C.fit_predict(X)
    elapsed_time = perf_counter() - t0

    del C
    gc.collect()

    return y_pred, elapsed_time

def fit_transform(X, p):
    t0 = perf_counter()
    X_new = p.fit_transform(X)
    elapsed_time = perf_counter() - t0
    return X_new, elapsed_time

def fit(X, p):
    t0 = perf_counter()
    p.fit(X)
    elapsed_time = perf_counter() - t0
    return p, elapsed_time

def fit_xy(X, y, p):
    t0 = perf_counter()
    p.fit(X, y)
    elapsed_time = perf_counter() - t0
    return p, elapsed_time

def project(X, p):
    t0 = perf_counter()
    X_new = p.transform(X)
    elapsed_time = perf_counter() - t0
    return X_new, elapsed_time

def invert(X_2d, p):
    t0 = perf_counter()
    X_new = p.inverse_transform(X_2d)
    elapsed_time = perf_counter() - t0
    return X_new, elapsed_time

def compute_all_metrics(X, X_2d, D_high, D_low, y, X_inv=None):
    T = metrics.metric_trustworthiness(X, X_2d, D_high, D_low)
    C = metrics.metric_continuity(X, X_2d, D_high, D_low)
    R = metrics.metric_shepard_diagram_correlation(D_high, D_low)
    S = metrics.metric_normalized_stress(D_high, D_low)
    N = metrics.metric_neighborhood_hit(X_2d, y)

    if X_inv is not None:
        MSE = metrics.metric_mse(X, X_inv)
    else:
        MSE = -99.0
    
    return T, C, R, S, N, MSE

def run_nonparam(dataset_name, test_name, p, X_train, y_train, X_test, y_test, D_train_high, D_test_high):
    X_train_2d, time_fit = fit_transform(X_train, p)
    X_test_2d = None

    time_train_pred = -99.0
    time_test_pred = -99.0

    time_train_inv = -99.0
    time_test_inv = -99.0

    X_train_nd = None
    X_test_nd = None

    D_train_2d = metrics.compute_distance_list(X_train_2d)
    D_test_2d = None

    return  X_train_2d, X_test_2d, X_train_nd, X_test_nd, D_train_2d, D_test_2d, time_fit, time_train_pred, time_test_pred, time_train_inv, time_test_inv

def run_param(dataset_name, test_name, p, X_train, y_train, X_test, y_test, D_train_high, D_test_high):
    p, time_fit = fit(X_train, p)
    X_train_2d, time_train_pred = project(X_train, p)
    X_test_2d, time_test_pred = project(X_test, p)

    time_train_inv = -99.0
    time_test_inv = -99.0

    X_train_nd = None
    X_test_nd = None

    D_train_2d = metrics.compute_distance_list(X_train_2d)
    D_test_2d = metrics.compute_distance_list(X_test_2d)

    return  X_train_2d, X_test_2d, X_train_nd, X_test_nd, D_train_2d, D_test_2d, time_fit, time_train_pred, time_test_pred, time_train_inv, time_test_inv

def run_param_inverse(dataset_name, test_name, p, X_train, y_train, X_test, y_test, D_train_high, D_test_high):
    p, time_fit = fit(X_train, p)
    X_train_2d, time_train_pred = project(X_train, p)
    X_test_2d, time_test_pred = project(X_test, p)

    X_train_nd, time_train_inv = invert(X_train_2d, p)
    X_test_nd, time_test_inv = invert(X_test_2d, p)

    D_train_2d = metrics.compute_distance_list(X_train_2d)
    D_test_2d = metrics.compute_distance_list(X_test_2d)

    return  X_train_2d, X_test_2d, X_train_nd, X_test_nd, D_train_2d, D_test_2d, time_fit, time_train_pred, time_test_pred, time_train_inv, time_test_inv

def run_xy(dataset_name, test_name, p, X_train, y_train, X_test, y_test, D_train_high, D_test_high, y_pseudo):
    p, time_fit = fit_xy(X_train, y_pseudo, p)
    X_train_2d, time_train_pred = project(X_train, p)
    X_test_2d, time_test_pred = project(X_test, p)

    time_train_inv = -99.0
    time_test_inv = -99.0

    X_train_nd = None
    X_test_nd = None

    D_train_2d = metrics.compute_distance_list(X_train_2d)
    D_test_2d = metrics.compute_distance_list(X_test_2d)

    return  X_train_2d, X_test_2d, X_train_nd, X_test_nd, D_train_2d, D_test_2d, time_fit, time_train_pred, time_test_pred, time_train_inv, time_test_inv

def run_xy_inverse(dataset_name, test_name, p, X_train, y_train, X_test, y_test, D_train_high, D_test_high, y_pseudo):
    p, time_fit = fit_xy(X_train, y_pseudo, p)
    X_train_2d, time_train_pred = project(X_train, p)
    X_test_2d, time_test_pred = project(X_test, p)

    X_train_nd, time_train_inv = invert(X_train_2d, p)
    X_test_nd, time_test_inv = invert(X_test_2d, p)

    D_train_2d = metrics.compute_distance_list(X_train_2d)
    D_test_2d = metrics.compute_distance_list(X_test_2d)

    return  X_train_2d, X_test_2d, X_train_nd, X_test_nd, D_train_2d, D_test_2d, time_fit, time_train_pred, time_test_pred, time_train_inv, time_test_inv

def run_projections(output_dir, dataset_name, test_names, projections, X_train, y_train, X_test, y_test, D_train_high, D_test_high, fit_type, lead_fit_time=0.0, y_pseudo=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {}
    metric_vals = []

    for test_name, p in zip(test_names, projections):
        print(test_name)

        if y_pseudo is not None and (len(np.unique(y_pseudo)) < 2 or -1 in np.unique(y_pseudo) or (test_name == 'LDA' and len(np.unique(y_pseudo)) <= 2)):
            print('{0} - {1}:# of classes <= 2 or class -1 present'.format(dataset_name, test_name))
            metric_vals.append((dataset_name,
                                test_name,
                                -99.0,  #time to fit
                                -99.0, #time to pred train
                                -99.0,  #time to pred test
                                -99.0,  #time to inv train
                                -99.0,   #time to inv test
                                -99.0,
                                -99.0,
                                -99.0,
                                -99.0,
                                -99.0,
                                -99.0,
                                -99.0,
                                -99.0,
                                -99.0,
                                -99.0,
                                -99.0,
                                -99.0))
            continue

        if fit_type.startswith('nonparam'):
            res = run_nonparam(dataset_name, test_name, p, X_train, y_train, X_test, y_test, D_train_high, D_test_high)
        elif fit_type.startswith('param'):
            res = run_param(dataset_name, test_name, p, X_train, y_train, X_test, y_test, D_train_high, D_test_high)
        elif fit_type.startswith('param_inverse'):
            res = run_param_inverse(dataset_name, test_name, p, X_train, y_train, X_test, y_test, D_train_high, D_test_high)
        elif fit_type.startswith('xy_direct'):
            res = run_xy(dataset_name, test_name, p, X_train, y_train, X_test, y_test, D_train_high, D_test_high, y_pseudo)
        elif fit_type.startswith('xy_inverse'):
            res = run_xy_inverse(dataset_name, test_name, p, X_train, y_train, X_test, y_test, D_train_high, D_test_high, y_pseudo)
        else:
            raise Exception('Invalid fit mode')

        X_train_2d = res[0]
        X_test_2d = res[1]
        X_train_nd = res[2]
        X_test_nd = res[3]
        D_train_2d = res[4]
        D_test_2d = res[5]
        time_fit = res[6]
        time_train_pred = res[7]
        time_test_pred = res[8]
        time_train_inv = res[9]
        time_test_inv = res[10]

        T_train, C_train, R_train, S_train, N_train, MSE_train = compute_all_metrics(X_train, X_train_2d, D_train_high, D_train_2d, y_train, X_inv=X_train_nd)
        plot(X_train_2d, y_train, figname=os.path.join(output_dir, '{0}_{1}_{2}.png'.format(dataset_name, test_name, 'train')))

        if X_test_2d is not None:
            T_test, C_test, R_test, S_test, N_test, MSE_test = compute_all_metrics(X_test, X_test_2d, D_test_high, D_test_2d, y_test, X_inv=X_test_nd)
            plot(X_test_2d, y_test, figname=os.path.join(output_dir, '{0}_{1}_{2}.png'.format(dataset_name, test_name, 'test')))
        else:
            T_test = -99.0
            C_test = -99.0
            R_test = -99.0
            S_test = -99.0
            N_test = -99.0
            MSE_test = -99.0

        results[test_name] = {}
        results[test_name]['X_train_2d'] = X_train_2d
        results[test_name]['X_test_2d'] = X_test_2d
        results[test_name]['X_train_nd'] = X_train_nd
        results[test_name]['X_test_nd'] = X_test_nd
        
        metric_vals.append((dataset_name,
                            test_name,
                            time_fit + lead_fit_time,  #time to fit
                            time_train_pred, #time to pred train
                            time_test_pred,  #time to pred test
                            time_train_inv,  #time to inv train
                            time_test_inv,   #time to inv test
                            T_train,
                            C_train,
                            R_train,
                            S_train,
                            N_train,
                            MSE_train,
                            T_test,
                            C_test,
                            R_test,
                            S_test,
                            N_test,
                            MSE_test))

    joblib.dump(results, os.path.join(output_dir, 'results_{0}_{1}.pkl'.format(dataset_name, fit_type)))

    df = pd.DataFrame(metric_vals, columns=['dataset_name',
                                            'test_name',
                                            'time_fit',
                                            'time_train_pred',
                                            'time_test_pred',
                                            'time_train_inv',
                                            'time_test_inv',
                                            'T_train',
                                            'C_train',
                                            'R_train',
                                            'S_train',
                                            'N_train',
                                            'MSE_train',
                                            'T_test',
                                            'C_test',
                                            'R_test',
                                            'S_test',
                                            'N_test',
                                            'MSE_test'])

    df.to_csv(os.path.join(output_dir, 'metrics_{0}_{1}.csv'.format(dataset_name, fit_type)), header=True, index=None)

def merge_metric_files(output_dir, output_file):
    files = glob(os.path.join(output_dir, 'metrics_*.csv'))

    dfs = []

    for f in files:
        df_temp = pd.read_csv(f)
        dfs.append(df_temp)

    df = pd.concat(dfs)
    df.to_csv(os.path.join(output_dir, output_file), index=None)

    for f in files:
        os.remove(f)

def resample(X, y, size):
    aug_factor = size / X.shape[0]

    if aug_factor <= 1.0:
        #subsample
        X_, _, y_, _ = train_test_split(X, y, train_size=aug_factor, random_state=420, stratify=y)
    else:
        #oversample
        X_ = np.repeat(X, np.ceil(aug_factor), axis=0)
        y_ = np.repeat(y, np.ceil(aug_factor), axis=0)
        X_ = X_[:size,:]
        y_ = y_[:size]

    return X_, y_


def save_timings(metric_vals, file_name):
    df = pd.DataFrame(metric_vals, columns=['dataset_name',
                                        'test_name',
                                        'train_size',
                                        'test_size',
                                        'time_fit',
                                        'time_test_pred',
                                        'time_test_inv'])

    df.to_csv(file_name, header=True, index=None)


def eval_projections_time(output_dir, dataset_name, test_names, projections, X, y, train_sizes, test_sizes, fit_type):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    metric_vals = []
    results_file_name = os.path.join(output_dir, 'metrics_time_{0}_{1}.csv'.format(dataset_name, fit_type))

    for test_name, p in zip(test_names, projections):
        first_training = True
        print(test_name)

        for train_size in train_sizes:
            print('Train size: {0}'.format(train_size))

            X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, train_size=0.5, random_state=420, stratify=y)
            X_train, y_train = resample(X_train_, y_train_, train_size)

            time_fit = -99.0
            time_test_pred = -99.0
            time_test_inv = -99.0
            test_size = -99.0

            try:
                # if nonparam, fit, transform and skip
                if fit_type == 'nonparam':
                    _, time_fit = fit_transform(X_train, p)
                #if xy, fit with labels
                elif fit_type.startswith('xy_'):
                    p, time_fit = fit_xy(X_train, y_train, p)
                #else do regular fit
                else:
                    p, time_fit = fit(X_train, p)

                #if p is parametric, run inference only for the first training
                if first_training and fit_type != 'nonparam':
                    for test_size in test_sizes:
                        print('Testing with {0}'.format(test_size))
                        X_test, _ = resample(X_test_, y_test_, test_size)

                        X_test_2d = None
                        X_test_nd = None

                        #try do to direct proj. On error set time to -99
                        try:
                            X_test_2d, time_test_pred = project(X_test, p)
                        except Exception as ex:
                            time_test_pred = -99.0

                        #try do to inverse proj. On error set time to -99
                        try:
                            if 'inverse' in fit_type:
                                X_test_nd, time_test_inv = invert(X_test_2d, p)
                            else:
                                time_test_inv = -99.0
                        except Exception as ex:
                            time_test_inv = -99.0

                        X_test = None
                        X_test_2d = None
                        X_test_nd = None

                        del X_test
                        del X_test_2d
                        del X_test_nd

                        metric_vals.append((dataset_name,
                            test_name,
                            train_size,
                            test_size,
                            time_fit,        #time to fit
                            time_test_pred,  #time to pred test
                            time_test_inv))  #time to inv test
                    
                    first_training = False
                else:
                    metric_vals.append((dataset_name,
                                        test_name,
                                        train_size,
                                        test_size,
                                        time_fit,        #time to fit
                                        time_test_pred,  #time to pred test
                                        time_test_inv))  #time to inv test


            #on fit error, record null result and skip
            except Exception as ex:
                print('ERROR: {0}'.format(str(ex)))
                metric_vals.append((dataset_name,
                                    test_name,
                                    train_size,
                                    test_size,
                                    -99.0,  #time to fit
                                    -99.0,  #time to pred test
                                    -99.0))  #time to inv test

            save_timings(metric_vals, results_file_name)

        gc.collect()
