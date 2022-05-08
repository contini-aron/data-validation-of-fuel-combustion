"""
concrete clustering algorythm : Affinity Propagation
"""
import pandas as pd
from src.clustering_algo.clustering_algorythm import ClusteringAlgorythm
from sklearn import cluster


class AlgoAffinityPropagation(ClusteringAlgorythm):
    """
    concrete clustering algorythm : Affinity Propagation
    """

    @staticmethod
    def cluster(df: pd.DataFrame, columns: list[str], **kwargs) -> pd.DataFrame:
        """
        :param df: the input dataframe
        :param columns: the columns on where to run HDBSCAN
        :return: the clustered Dataframe
        """
        to_cluster = df.loc[:, columns]
        # for i in range(1,100):
        clusters = cluster.AffinityPropagation(
            random_state=5, damping=0.9, max_iter=1000
        ).fit(to_cluster)
        retval = df.copy()
        retval["ClusterID"] = clusters.labels_
        return retval
