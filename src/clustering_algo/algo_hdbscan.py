"""
Concrete implementation of clustering algorythm following HDBSCAN algorythm
"""

import hdbscan
import pandas as pd
from clustering_algo.clustering_algorythm import ClusteringAlgorythm


class AlgoHDBSCAN(ClusteringAlgorythm):
    """
    the concrete algorythm
    """

    @staticmethod
    def cluster(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        :param df: the input dataframe
        :param columns: the columns on where to run HDBSCAN
        :return: the clustered Dataframe
        """
        to_cluster = df.loc[:, columns]
        n_cluster, prob, n_noise, best_metric = (
            "n_cluster",
            "probabilty",
            "n_noise",
            "best_metric",
        )
        best_min_cluster = {
            n_cluster: 0,
            prob: 0,
            n_noise: to_cluster.shape[0],
            best_metric: "euclidean",
        }
        for metrics in hdbscan.dist_metrics.METRIC_MAPPING.keys():
            for i in range(2, min(100, to_cluster.shape[0])):
                clusterer = hdbscan.HDBSCAN(min_cluster_size=i)
                clusterer = clusterer.fit(to_cluster)
                if (
                    clusterer.probabilities_[clusterer.probabilities_ == 0].shape[0]
                    <= best_min_cluster[n_noise]
                ):
                    if (
                        clusterer.probabilities_[clusterer.probabilities_ > 0].mean()
                        > best_min_cluster[prob]
                    ):
                        best_min_cluster[n_cluster] = i
                        best_min_cluster[prob] = clusterer.probabilities_[
                            clusterer.probabilities_ > 0
                        ].mean()
                        best_min_cluster[n_noise] = clusterer.probabilities_[
                            clusterer.probabilities_ == 0
                        ].shape[0]
                        if metrics != best_min_cluster[best_metric]:
                            best_min_cluster[best_metric] = metrics
        # probabilities = str(best_min_cluster)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=best_min_cluster[n_cluster],
            metric=best_min_cluster[best_metric],
        )
        clusterer = clusterer.fit(to_cluster)
        retval = df.copy()
        retval["ClusterID"] = clusterer.labels_
        return retval
