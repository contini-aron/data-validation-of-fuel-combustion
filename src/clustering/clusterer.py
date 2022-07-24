"""
cluster the data and retrieve the matrix containing the clusters
"""
from __future__ import annotations
from importlib.metadata import metadata
import pandas as pd
from . import ClusteringAlgorythm
from .grouped import groupanddescribe
from .utils.percentile import get_percentile_density


class Clusterer:
    """
    class that provides clustered data once an algorythm is chosen
    """

    clustered: pd.DataFrame
    metadata: pd.DataFrame
    percentiles: float
    best_clusters: list[int]
    groupby_clustered : dict = {}

    def __init__(self, algorythm: ClusteringAlgorythm) -> None:
        self.clustering_algo = algorythm

    @property
    def algorythm(self) -> ClusteringAlgorythm:
        """

        :return: the currently saved clustering algorythm
        """
        return self.clustering_algo

    @algorythm.setter
    def algorythm(self, algorythm: ClusteringAlgorythm) -> None:
        """
        method to change clustering algorythm
        :param algorythm:
        """
        self.clustering_algo = algorythm

    def start(self, df: pd.DataFrame, columns: list[str], groupby:list[str] = None, min_cluster_size:int = 5) -> pd.DataFrame:
        """
        start clustering algorythm
        :param columns: columns that will be used in clustering algo
        :param df: the input data
        :return: the clustered dataframe
        """
        if not groupby is None:
            ignored :list[str] = []
            for group in df.groupby(groupby):
                data = group[1]
                group_name = group[0]
                if data.shape[0]>=min_cluster_size:
                    print(f"working on GROUP:\t{group_name}")
                    self.groupby_clustered[group_name] = self.clustering_algo.cluster(df=data, columns=columns)
                else:
                    ignored.append(group_name)
            print(f"ignored groups are:{ignored}")


        else:
            self.clustered = self.clustering_algo.cluster(df=df, columns=columns)
            return self.clustered

    def metadata(
        self,
        data: pd.DataFrame = None,
        groupby_elem: list[str] = None,
        slc: list[str] = None,
        statistics_names: dict = None,
        count: bool = False,
    ) -> pd.DataFrame:
        """
        start metadata creation
        :param data: dataframe to procude dataframe on, if None self.clustered will be used
        :param groupby_elem: a list of the column to group by, if None ClusterID will be used
        :param statistics: a list of statisctics to be shown if None mean will be used
        :param count: a boolean if True the count will be shown

        :return: a Multiindex dataframe with all the stats
        """
        # creates a dictionary of str : function with the statistics
        stats = {
            stat_name: getattr(pd.core.groupby.GroupBy, stat_name)
            for stat_name in statistics_names
        }

        if data is None:
            data = self.clustered

        self.metadata = groupanddescribe(data, groupby_elem, slc, stats, count)
        return self.metadata

    def get_best_clusters(
        self,
        metadata=None,
        min_n_element: int = 5,
        std_threshold: float = 0.2,
        std_column: str = "Score",
        percentile: float = 0.25,
        percentile_stat: str = "mean",
        percentile_column: str = "Score",
    ) -> pd.DataFrame:
        """
        retrieves a metadata Multiindex DataFrame containing only the best clusters, rated by some input thresholds
        """
        # print(self.metadata["mean"])
        # print(self.metadata["mean"].loc[self.metadata["mean"]["Score"]>0, ["Score"]])

        # check if percentile is a percentage
        if not 0 < percentile < 1:
            raise Exception("given percentile is not a percentage")

        # check data
        if metadata is None:
            metadata = self.metadata
        # filter mean Score <=0
        metadata = metadata[metadata[percentile_stat][percentile_column] > 0]
        print(metadata[percentile_stat, percentile_column])

        # get nth percentile
        self.percentile = metadata[percentile_stat, percentile_column].quantile(
            percentile
        )
        print(
            f"the {int(percentile*100)}th percentile of {percentile_column} is {self.percentile}"
        )

        # filter only the rows that are below the percentile
        metadata = metadata[
            metadata[percentile_stat, percentile_column] < self.percentile
        ]

        # filter only the rows with low std
        metadata = metadata[metadata["std", std_column] < std_threshold]

        # filter only the clusters with at least min_n_element
        metadata = metadata[metadata["count"] >= min_n_element]
        print(f"best graded clusters are \n", *metadata.index)
        return metadata
