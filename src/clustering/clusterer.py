"""
cluster the data and retrieve the matrix containing the clusters
"""
from __future__ import annotations
import pandas as pd
from . import ClusteringAlgorythm
from .grouped import groupanddescribe, mkdir
from . import get_columns
import os
import matplotlib.pyplot as plt
from .clustering_algorythms.norm import normalize
import numpy as np
import plotly.express as px


class Clusterer:
    """
    class that provides clustered data once an algorythm is chosen
    """

    clustered: pd.DataFrame
    metadata: pd.DataFrame
    percentiles: float
    best_clusters: list[int]
    groupby_clustered: dict = {}
    filtered_metadata: pd.DataFrame
    grouped_by: list[str]

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

    def start(
        self,
        df: pd.DataFrame,
        columns: list[str],
        groupby: list[str] = None,
        min_cluster_size: int = 5,
    ) -> pd.DataFrame:
        """
        start clustering algorythm
        :param columns: columns that will be used in clustering algo
        :param df: the input data
        :return: the clustered dataframe
        """
        if not groupby is None:
            self.grouped_by = groupby
            ignored: list[str] = []
            for group in df.groupby(groupby):
                data = group[1]
                group_name = group[0]
                if data.shape[0] >= min_cluster_size:
                    print(f"working on GROUP:\t{group_name}")
                    self.groupby_clustered[group_name] = self.clustering_algo.cluster(
                        df=data.copy(), columns=columns
                    )
                else:
                    ignored.append(group_name)
            print(f"ignored groups are:{ignored}")

        else:
            self.clustered = self.clustering_algo.cluster(df=df, columns=columns)
            return self.clustered.copy()

    def get_metadata(
        self,
        data: pd.DataFrame = None,
        groupby_elem: list[str] = None,
        slc: list[str] = None,
        statistics_names: dict = None,
        count: bool = True,
    ) -> pd.DataFrame:
        """
        start metadata creation
        :param data: dataframe to procude dataframe on, if None self.clustered will be used
        :param groupby_elem: a list of the column to group by, if None ClusterID will be used
        :param statistics: a list of statisctics to be shown if None mean will be used
        :param count: a boolean if True the count will be shown

        :return: a Multiindex dataframe with all the stats
        """
        if statistics_names is None:
            statistics_names = ["mean", "min", "max", "median", "std", "var"]
        # creates a dictionary of str : function with the statistics
        stats = {
            stat_name: getattr(pd.core.groupby.GroupBy, stat_name)
            for stat_name in statistics_names
        }

        if data is None:
            data = self.clustered
        else:
            self.clustered = data

        if slc is None:
            slc = get_columns()

        self.metadata = groupanddescribe(data, groupby_elem, slc, stats, count)
        return self.metadata.copy()

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

        # check if percentile is a percentage
        if not 0 < percentile < 1:
            raise Exception("given percentile is not a percentage")

        # check data
        if metadata is None:
            metadata = self.metadata
        # filter mean Score <=0
        # print(metadata[percentile_stat, percentile_column])
        metadata = metadata[metadata[percentile_stat][percentile_column] > 0]

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
        self.best_clusters = [i for i in list(metadata.index) if i > 0]
        self.filtered_metadata = metadata
        return metadata.copy()

    def export(self, groupby: bool = False):
        meta_path = os.path.join(os.curdir, "metadata")
        mkdir(meta_path)

        if not groupby:
            whole_dataset_path = os.path.join(meta_path, "whole_dataset")
            mkdir(whole_dataset_path)
            self.metadata.to_excel(os.path.join(whole_dataset_path, "metadata.xlsx"))
            self.clustered.to_excel(os.path.join(whole_dataset_path, "clustered.xlsx"))
        else:
            for group_name, group in self.groupby_clustered.items():
                print("exporting group:", group_name)
                groups_path = os.path.join(meta_path, "grouped")
                mkdir(groups_path)
                group_by_path = os.path.join(
                    groups_path, f"grouped_by{self.grouped_by}"
                )
                mkdir(group_by_path)
                group_name = str(group_name).replace("'", "").replace('"', '')
                group_path = os.path.join(group_by_path, f"{group_name}")
                mkdir(group_path)
                group.to_excel(os.path.join(group_path, f"{group_name}clustered.xlsx"))
                self.get_metadata(group).to_excel(
                    os.path.join(group_path, f"{group_name}metadata.xlsx")
                )

    def parallel_coordinates_plot(
        self,
        df: pd.DataFrame = None,
        columns: list[str] = None, # if u want to specify the columns to consider
        cluster_id: str = "ClusterID", # the name of the cluster id column
        palette=[
            "#b30000",
            "#7c1158",
            "#4421af",
            "#1a53ff",
            "#0d88e6",
            "#00b7c7",
            "#5ad45a",
            "#8be04e",
            "#ebdc78",
        ],
        folder = None
    ):
        if columns is None:
            columns = [
                "Score",
                "d0L2",
                "d1L2",
                "d0Pe",
                "d1Pe",
                "Temperature0",
                "Temperature1",
                "Pressure0",
                "Pressure1",
            ]
        if df is None:
            to_plot = self.clustered[
                self.clustered["ClusterID"].isin(self.best_clusters)
            ].copy()
        else:
            to_plot = df.copy()
        to_plot = to_plot.loc[:, [i for i in columns] + ["ClusterID"]].copy()
        clusters = to_plot.loc[:, cluster_id].copy().to_numpy()
        fig = px.parallel_coordinates(
            to_plot,
            color=cluster_id,
            dimensions=columns,
            color_continuous_scale=palette,
        )
        if folder is None:
            parallel_folder = f"{os.curdir}{os.sep}metadata{os.sep}whole_dataset{os.sep}parallel_coordinates_plots"
        else:
            parallel_folder =  folder
        mkdir(parallel_folder)
        # fig.set_size_inches(40, 20)
        fig.write_html(f"{parallel_folder}{os.sep}best_clusters.html")
        plt.close()

        for index, cluster in enumerate(np.unique(clusters)):
            fig = px.parallel_coordinates(
                to_plot.loc[to_plot[cluster_id] == cluster],
                color=cluster_id,
                dimensions=columns,
                color_continuous_scale=palette,
            )
            fig.write_html(f"{parallel_folder}{os.sep}cluster{cluster}.html")

            plt.close()


    def groupby_parallel_plots(self, groupby_columns:list[str] = ["Target"], best:bool=False):
        groupby_folder = f"{os.curdir}{os.sep}metadata{os.sep}grouped{os.sep}grouped_by{groupby_columns}"
        # print(groupby_folder)
        for _, directories, _ in os.walk(groupby_folder):
            for directory in directories:
                if directory == "parallel_coordinates_plot":
                    continue
                print(f"plotting directory: {directory}")
                group_dir = f"{groupby_folder}{os.sep}{directory}"
                to_plot = pd.read_excel(f"{group_dir}{os.sep}{directory}clustered.xlsx")
                metadata = pd.read_excel(f"{group_dir}{os.sep}{directory}metadata.xlsx", index_col=[0], header=[0,1])
                count = metadata["count"].to_numpy().copy()
                metadata.drop(columns=["count"], inplace=True)
                metadata["count"] = count
                if best:
                    self.__set_metadata(metadata)
                    self.__set_clustered(to_plot)
                    self.get_best_clusters()
                self.parallel_coordinates_plot(folder=f"{group_dir}{os.sep}parallel_coordinates_plot")
    
    def __set_metadata(self, df:pd.DataFrame)->None:
        self.metadata = df
    def __set_clustered(self, df:pd.DataFrame)->None:
        self.clustered = df




