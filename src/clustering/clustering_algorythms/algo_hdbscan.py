"""
Concrete implementation of clustering algorythm following HDBSCAN algorythm
"""

import hdbscan
import pandas as pd
from . import ClusteringAlgorythm
from .norm import normalize


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
        to_cluster = normalize(to_cluster)
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

        # performing evaluation based on mean probability of fitting in a certain cluster
        # to find the best metrics and the best number of clusters
        # HDBSCAN is self evaluating and probabilty of fitting based, with noise detection
        for (
            metrics
        ) in (
            hdbscan.dist_metrics.METRIC_MAPPING.keys()
        ):  # loops in each metrics e.g euclidean , ...

            # loops in a number of cluster starting from 2 and going to the min between 100 and the number of rows in input
            for i in range(2, min(100, to_cluster.shape[0])):
                clusterer = hdbscan.HDBSCAN(min_cluster_size=i)
                clusterer = clusterer.fit(to_cluster)
                # more mean probability -> more likely fitting clusters
                if (
                    clusterer.probabilities_[clusterer.probabilities_ > 0].mean()
                    > best_min_cluster[prob]
                ):
                    # if a probability is 0 then belongs to a noise so we want to reduce it
                    if (
                        clusterer.probabilities_[clusterer.probabilities_ == 0].shape[0]
                        <= best_min_cluster[n_noise]
                    ):
                        # marks the clustering inputs as the best till now
                        best_min_cluster[n_cluster] = i
                        best_min_cluster[prob] = clusterer.probabilities_[
                            clusterer.probabilities_ > 0
                        ].mean()
                        best_min_cluster[n_noise] = clusterer.probabilities_[
                            clusterer.probabilities_ == 0
                        ].shape[0]
                        if metrics != best_min_cluster[best_metric]:
                            best_min_cluster[best_metric] = metrics

        # computes again for the best inputs
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=best_min_cluster[n_cluster],
            metric=best_min_cluster[best_metric],
        )
        clusterer = clusterer.fit(to_cluster)
        retval = df.copy()

        # adds a column defining with cluster the row belongs to
        retval["ClusterID"] = clusterer.labels_
        return retval
