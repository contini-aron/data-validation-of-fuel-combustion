"""
boy
"""
import pandas as pd
import input_parse as ip
from src.clusterer import Clusterer
from clustering_algo.algo_hdbscan import HDBSCAN

if __name__ == "__main__":
    df = pd.read_excel("./input_files/data.xlsx")
    df = ip.parse(df)
    columns = [
        # "Experiment DOI",
        # "Exp SciExpeM ID",
        # "Chem Model",
        # "Chem Model ID",
        # "Experiment Type",
        # "Reactor",
        # "Target",
        # "Fuels",
        # "Phi",
        "Phi0",
        "Phi1",
        # "Pressure",
        "P0",
        "P1",
        # "Temperature",
        "T0",
        "T1",
        "Score",
        "Error",
        "d0L2",
        "d1L2",
        "d0Pe",
        "d1Pe",
        "shift"
    ]

    print(Clusterer(HDBSCAN).start(df, columns))
