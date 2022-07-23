"""
common Algorythm interface for clustering
"""
from __future__ import annotations

# importing abstract class
from abc import ABC, abstractmethod

import pandas as pd


class ClusteringAlgorythm(ABC):
    """
    common interface of clustering algo
    """

    @abstractmethod
    def cluster(self, data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        cluster the dataframe in input
        :param columns: column names that will be used in clustering
        :param data: pd.Dataframe containing data to cluster
        :return: pd.Dataframe containing a new column that indicates the cluster
        """
