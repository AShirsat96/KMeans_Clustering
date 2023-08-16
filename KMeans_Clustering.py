# -*- coding: utf-8 -*-
"""
@author: Aniket
"""

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sklearn.metrics.cluster as metrics
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import completeness_score, v_measure_score

data = pickle.load(open("data/documents.p", "rb"))

def cluster_articles(data):
    # Extract document vectors and group information from the data dictionary
    vectors = np.array(data['vectors'])
    groups = np.array(data['group'])

    # Preprocess the data using Z-Score normalization
    scaler = StandardScaler()
    normalized_vectors = scaler.fit_transform(vectors)

    # Perform KMeans clustering on the original normalized document vectors
    kmeans_original = KMeans(n_clusters=18, random_state=2, tol=8.85, max_iter=58)
    original_clusters = kmeans_original.fit_predict(normalized_vectors)

    # Perform PCA to reduce dimensionality of the normalized document vectors
    pca = PCA(n_components=10, random_state=2)
    reduced_vectors = pca.fit_transform(normalized_vectors)

    # Perform KMeans clustering on the reduced normalized vectors
    kmeans_reduced = KMeans(n_clusters=18, random_state=2, tol=8.85, max_iter=58)
    reduced_clusters = kmeans_reduced.fit_predict(reduced_vectors)

    # Calculate metrics
    nobs_100 = [np.sum(original_clusters == i) for i in range(18)]
    nobs_10 = [np.sum(reduced_clusters == i) for i in range(18)]
    pca_explained = pca.explained_variance_ratio_[0]
    cs_100 = completeness_score(groups, original_clusters)
    cs_10 = completeness_score(groups, reduced_clusters)
    vms_100 = v_measure_score(groups, original_clusters)
    vms_10 = v_measure_score(groups, reduced_clusters)

    # Create a dictionary to store the metrics
    metrics_dict = {
        'nobs_100': nobs_100,
        'nobs_10': nobs_10,
        'pca_explained': pca_explained,
        'cs_100': cs_100,
        'cs_10': cs_10,
        'vms_100': vms_100,
        'vms_10': vms_10,
    }

    return metrics_dict
