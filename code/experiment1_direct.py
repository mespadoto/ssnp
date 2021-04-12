#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
from PIL import Image, ImageDraw, ImageFont
from skimage import io
from sklearn.cluster import (DBSCAN, AffinityPropagation,
                             AgglomerativeClustering, Birch, KMeans)
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP

import ae
import metrics
import ssnp
import nnproj

def compute_all_metrics(X, X_2d, D_high, D_low, y, X_inv=None):
    T = metrics.metric_trustworthiness(X, X_2d, D_high, D_low)
    C = metrics.metric_continuity(X, X_2d, D_high, D_low)
    R = metrics.metric_shepard_diagram_correlation(D_high, D_low)
    S = metrics.metric_normalized_stress(D_high, D_low)
    N = metrics.metric_neighborhood_hit(X_2d, y, k=3)

    if X_inv is not None:
        MSE = metrics.metric_mse(X, X_inv)
    else:
        MSE = -99.0
    
    return T, C, R, S, N, MSE

def plot(X, y, figname=None):
    if len(np.unique(y)) <= 10:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = plt.get_cmap('tab20')

    fig, ax = plt.subplots(figsize=(20, 20))
    
    for cl in np.unique(y):
        ax.scatter(X[y==cl,0], X[y==cl,1], c=[cmap(cl)], label=cl, s=20)
        ax.axis('off')

    if figname is not None:
        fig.savefig(figname)

    plt.close('all')
    del fig
    del ax


if __name__ == '__main__':
    patience = 5
    epochs = 200
    
    min_delta = 0.05

    verbose = False
    results = []

    output_dir = 'results_direct'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir ='../data'
    data_dirs = ['mnist', 'fashionmnist', 'har', 'reuters']

    epochs_dataset = {}
    epochs_dataset['fashionmnist'] = 10
    epochs_dataset['mnist'] = 10
    epochs_dataset['har'] = 10
    epochs_dataset['reuters'] = 10

    classes_mult = {}
    classes_mult['fashionmnist'] = 2
    classes_mult['mnist'] = 2
    classes_mult['har'] = 2
    classes_mult['reuters'] = 1

    for d in data_dirs:
        dataset_name = d

        X = np.load(os.path.join(data_dir, d, 'X.npy'))
        y = np.load(os.path.join(data_dir, d, 'y.npy'))

        print('------------------------------------------------------')
        print('Dataset: {0}'.format(dataset_name))
        print(X.shape)
        print(y.shape)
        print(np.unique(y))

        n_classes = len(np.unique(y)) * classes_mult[dataset_name]
        n_samples = X.shape[0]

        train_size = min(int(n_samples*0.9), 5000)

        X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size, random_state=420, stratify=y)
        D_high = metrics.compute_distance_list(X_train)

        epochs = epochs_dataset[dataset_name]

        ssnpgt = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear')
        ssnpgt.fit(X_train, y_train)
        X_ssnpgt = ssnpgt.transform(X_train)

        ssnpkm = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear')
        C = KMeans(n_clusters=n_classes)
        y_km = C.fit_predict(X_train)
        ssnpkm.fit(X_train, y_km)
        X_ssnpkm = ssnpkm.transform(X_train)

        ssnpag = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear')
        C = AgglomerativeClustering(n_clusters=n_classes)
        y_ag = C.fit_predict(X_train)
        ssnpag.fit(X_train, y_ag)
        X_ssnpag = ssnpag.transform(X_train)

        tsne = TSNE(n_jobs=4, random_state=420)
        X_tsne = tsne.fit_transform(X_train)

        ump = UMAP(random_state=420)
        X_umap = ump.fit_transform(X_train)

        aep = ae.AutoencoderProjection(epochs=epochs, verbose=0)
        aep.fit(X_train)
        X_aep = aep.transform(X_train)

        nnp = nnproj.NNProj(init=TSNE(n_jobs=4, random_state=420))
        nnp.fit(X_train)
        X_nnp = nnp.transform(X_train)

        D_ssnpgt = metrics.compute_distance_list(X_ssnpgt)
        D_ssnpkm = metrics.compute_distance_list(X_ssnpkm)
        D_ssnpag = metrics.compute_distance_list(X_ssnpag)
        D_tsne = metrics.compute_distance_list(X_tsne)
        D_umap = metrics.compute_distance_list(X_umap)
        D_aep = metrics.compute_distance_list(X_aep)
        D_nnp = metrics.compute_distance_list(X_nnp)

        results.append((dataset_name, 'SSNP-GT',) + compute_all_metrics(X_train, X_ssnpgt, D_high, D_ssnpgt, y_train))
        results.append((dataset_name, 'SSNP-KMeans',) + compute_all_metrics(X_train, X_ssnpkm, D_high, D_ssnpkm, y_train))
        results.append((dataset_name, 'SSNP-AG',) + compute_all_metrics(X_train, X_ssnpag, D_high, D_ssnpag, y_train))
        results.append((dataset_name, 'AE',) + compute_all_metrics(X_train, X_aep, D_high, D_aep, y_train))
        results.append((dataset_name, 'TSNE',) + compute_all_metrics(X_train, X_tsne, D_high, D_tsne, y_train))
        results.append((dataset_name, 'UMAP',) + compute_all_metrics(X_train, X_umap, D_high, D_umap, y_train))
        results.append((dataset_name, 'NNP',) + compute_all_metrics(X_train, X_nnp, D_high, D_nnp, y_train))

        for X_, label in zip([X_ssnpgt, X_ssnpkm, X_ssnpag, X_umap, X_tsne, X_aep, X_nnp], ['SSNP-GT', 'SSNP-KMeans', 'SSNP-AG', 'UMAP', 'TSNE', 'AE', 'NNP']):
            fname = os.path.join(output_dir, '{0}_{1}.png'.format(dataset_name, label))
            print(fname)
            plot(X_, y_train, fname)

    df = pd.DataFrame(results, columns=[    'dataset_name',
                                            'test_name',
                                            'T_train',
                                            'C_train',
                                            'R_train',
                                            'S_train',
                                            'N_train',
                                            'MSE_train'])

    df.to_csv(os.path.join(output_dir, 'metrics.csv'), header=True, index=None)

    #don't plot NNP
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 50)
    pri_images = ['SSNP-KMeans', 'SSNP-AG', 'AE', 'TSNE', 'UMAP', 'SSNP-GT']

    images = glob(output_dir + '/*.png')    
    base = 2000

    for d in data_dirs:
        dataset_name = d
        to_paste = []    

        for i, label in enumerate(pri_images):
            to_paste += [f for f in images if os.path.basename(f) == '{0}_{1}.png'.format(dataset_name, label)]

        img = np.zeros((base, base*6, 3)).astype('uint8')
        
        for i, im in enumerate(to_paste):
            tmp = io.imread(im)
            img[:,i*base:(i+1)*base,:] = tmp[:,:,:3]

        pimg = Image.fromarray(img)
        pimg.save(output_dir + '/composite_full_{0}.png'.format(dataset_name))

        for i, label in enumerate(pri_images):
            print('/composite_full_{0}.png'.format(dataset_name), "{0} {1}".format(dataset_name, label))


    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 50)
    pri_images = ['SSNP-KMeans', 'SSNP-AG', 'AE']

    images = glob(output_dir + '/*.png')    
    base = 2000

    for d in data_dirs:
        dataset_name = d
        to_paste = []    

        for i, label in enumerate(pri_images):
            to_paste += [f for f in images if os.path.basename(f) == '{0}_{1}.png'.format(dataset_name, label)]

        img = np.zeros((base, base*3, 3)).astype('uint8')
        
        for i, im in enumerate(to_paste):
            tmp = io.imread(im)
            img[:,i*base:(i+1)*base,:] = tmp[:,:,:3]

        pimg = Image.fromarray(img)
        pimg.save(output_dir + '/composite_{0}.png'.format(dataset_name))

        for i, label in enumerate(pri_images):
            print('/composite_{0}.png'.format(dataset_name), "{0} {1}".format(dataset_name, label))
