"""
concrete clustering algorythm : KMEANS
"""
from math import sqrt
import pandas as pd
from . import ClusteringAlgorythm
from sklearn import cluster
from .norm import normalize


def get_n_best(df: pd.DataFrame):
    """
    retrieves the best number of cluster to use using inertia and elbow method
    """
    distortions = []
    max_clusters = min(100, df.shape[0])
    for i in range(2, max_clusters):
        kmeans = cluster.KMeans(n_clusters=i)
        kmeans.fit(df)
        distortions.append(kmeans.inertia_)

    # using the point to line distance formula to get the elbow point
    # max distance point to line (between first and last point) -> elbow point

    # assign the first point
    x1, y1 = 2, distortions[0]

    # assign last point
    x2, y2 = max_clusters, distortions[-1]

    distances = []
    for index, y0 in enumerate(distortions):
        x0 = index + 2  # the first point is 2
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator / denominator)
    return distances.index(max(distances)) + 2 # the first n_cluster is 2


class AlgoKmeans(ClusteringAlgorythm):
    """
    concrete clustering algorythm : KMEANS
    """

    @staticmethod
    def cluster(df: pd.DataFrame, columns: list[str], **kwargs) -> pd.DataFrame:
        """
        :param df: the input dataframe
        :param columns: the columns on where to run HDBSCAN
        :return: the clustered Dataframe
        """
        print("evaluating algorythm Kmeans")
        to_cluster = df.loc[:, columns]
        to_cluster = normalize(to_cluster)
        n_best = get_n_best(to_cluster)
        print("best n of cluster is", n_best)
        clusters = cluster.KMeans(n_clusters=n_best).fit(to_cluster)
        retval = df.copy()
        retval["ClusterID"] = clusters.labels_ + 1 # to maintain the count from 1
        return retval