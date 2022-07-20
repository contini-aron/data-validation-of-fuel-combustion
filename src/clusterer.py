"""
cluster the data and retrieve the matrix containing the clusters
"""
from __future__ import annotations
import pandas as pd
from input_parse import parse
from clustering_algo.clustering_algorythm import ClusteringAlgorythm


class Clusterer:
    """
    class that provides clustered data once an algorythm is chosen
    """

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

    def start(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        start clustering algorythm
        :param columns: columns that will be used in clustering algo
        :param df: the input data
        :return: the clustered dataframe
        """
        return self.clustering_algo.cluster(df=df, columns=columns)


if __name__ == "__main__":
    input_data = pd.read_excel(
        "/home/aron/Documents/Progetto_Data_mining/"
        "data-validation-of-fuel-combustion/src/input_files/data.xlsx"
    )
    input_data = parse(input_data)
