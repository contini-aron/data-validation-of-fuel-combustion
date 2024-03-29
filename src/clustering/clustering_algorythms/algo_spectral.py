"""
credits for the self clustering evaluation implementation: https://towardsdatascience.com/spectral-graph-clustering-and-optimal-number-of-clusters-estimation-32704189afbe
"""

import pandas as pd
from . import ClusteringAlgorythm
from sklearn import cluster
from .norm import normalize
import scipy
from scipy.sparse import csgraph
# from scipy.sparse.linalg import eigsh
from numpy import linalg as la
import numpy as np
from scipy.spatial.distance import pdist, squareform

def getAffinityMatrix(coordinates, k = 7):
    """
    Calculate affinity matrix based on input coordinates matrix and the number
    of nearest neighbours.
    
    Apply local scaling based on the k nearest neighbour
        References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    """
    # calculate euclidian distance matrix
    dists = squareform(pdist(coordinates)) 
    
    # for each row, sort the distances ascendingly and take the index of the 
    #k-th position (nearest neighbour)
    knn_distances = np.sort(dists, axis=0)[k]
    knn_distances = knn_distances[np.newaxis].T
    
    # calculate sigma_i * sigma_j
    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale
    # divide square distance matrix by local scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix

def eigenDecomposition(A, topK = 1):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return: the optimal number of clusters by eigengap heuristic
    
    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic
    
    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]
    
    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in 
    # the euclidean norm of complex numbers.
    # eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = la.eig(L)
    
    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1
        
    return int(nb_clusters)

class AlgoSpectral(ClusteringAlgorythm):
    """
    concrete clustering algorythm : Spectral Clustering
    """

    @staticmethod
    def cluster(df: pd.DataFrame, columns: list[str], **kwargs) -> pd.DataFrame:
        """
        :param df: the input dataframe
        :param columns: the columns on where to run Spectral
        :return: the clustered Dataframe
        """
        print("evaluating algorythm Spectral")
        to_cluster = df.loc[:, columns]
        to_cluster = normalize(to_cluster)
        aff_matrix = getAffinityMatrix(to_cluster, k=min(to_cluster.shape[0]-1, 7))
        n_clusters = eigenDecomposition(aff_matrix)
        print("best number of clusters is ", n_clusters)
        clusters = cluster.SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0).fit(to_cluster)
        retval = df.copy()
        retval["ClusterID"] = clusters.labels_ + 1 # to maintain the count from 1
        return retval