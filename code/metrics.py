#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy import spatial
from scipy import stats
from scipy import optimize

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def compute_distance_list(X):
    return spatial.distance.pdist(X, 'euclidean')

def metric_neighborhood_hit(X, y, k=7):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    neighbors = knn.kneighbors(X, return_distance=False)
    return np.mean(np.mean((y[neighbors] == np.tile(y.reshape((-1, 1)), k)).astype('uint8'), axis=1))

def metric_trustworthiness(X_high, X_low, D_high_m, D_low_m, k=7):
    D_high = spatial.distance.squareform(D_high_m)
    D_low  = spatial.distance.squareform(D_low_m)

    n = X_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])

        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())


def metric_continuity(X_high, X_low, D_high_l, D_low_l, k=7):
    D_high = spatial.distance.squareform(D_high_l)
    D_low  = spatial.distance.squareform(D_low_l)

    n = X_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())

def metric_shepard_diagram_correlation(D_high, D_low):
    return stats.spearmanr(D_high, D_low)[0]


def metric_normalized_stress(D_high, D_low):
    return np.sum((D_high - D_low)**2) / np.sum(D_high**2)

def metric_mse(X, X_hat):
    return np.mean(np.square(X - X_hat))
