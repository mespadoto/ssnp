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

    output_dir = 'results_clustering'

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

        train_size = 5000
        test_size = 1000

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=420, stratify=y)

        epochs = epochs_dataset[dataset_name]

        ssnpgt = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear')
        ssnpgt.fit(X_train, y_train)
        X_ssnpgt = ssnpgt.transform(X_train)
        y_pred = ssnpgt.predict(X_train)
        y_pred_test = ssnpgt.predict(X_test)

        ssnpkm = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear')
        C = KMeans(n_clusters=n_classes)
        y_km = C.fit_predict(X_train)
        ssnpkm.fit(X_train, y_km)
        X_ssnpkm = ssnpkm.transform(X_train)
        y_pred_km = ssnpkm.predict(X_train)
        
        y_pred_km_test_cl = C.predict(X_test)
        y_pred_km_test = ssnpkm.predict(X_test)

        results.append((dataset_name, 'SSNP-GT', 'train', np.mean(y_train == y_pred)))
        results.append((dataset_name, 'SSNP-KMeans', 'train', np.mean(y_km == y_pred_km)))

        results.append((dataset_name, 'SSNP-GT', 'test', np.mean(y_test == y_pred_test)))
        results.append((dataset_name, 'SSNP-KMeans', 'test', np.mean(y_pred_km_test_cl == y_pred_km_test)))


    df = pd.DataFrame(results, columns=[    'dataset_name',
                                            'test_name',
                                            'type',
                                            'acc'])

    df.to_csv(os.path.join(output_dir, 'metrics.csv'), header=True, index=None)
