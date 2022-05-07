"""
cluster the data and retrieve the matrix containing the clusters
"""
import pandas as pd
from input_parse import parse

if __name__ == "__main__":
    input_data = pd.read_excel(
        "/home/aron/Documents/Progetto_Data_mining/"
        "data-validation-of-fuel-combustion/input_files/data.xlsx"
    )
    input_data = parse(input_data)
