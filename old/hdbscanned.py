import hdbscan
import pandas as pd
import ast
import numpy as np

data = pd.read_excel("data.xlsx", index_col=0)
data.head()

# rename indexes

# +
data.rename(
    columns={"Pressure (Bar)": "Pressure", "Temperature (K)": "Temperature"},
    inplace=True,
)

data.head()
# -

# transform Phi, Temperature(K) column into a column of tuples

data["Phi"] = data["Phi"].apply(ast.literal_eval)
data["Temperature"] = data["Temperature"].apply(ast.literal_eval)
data["Temperature Symbol"] = "K"
data["Pressure"] = data["Pressure"].apply(ast.literal_eval)
data["Pressure Symbol"] = "Bar"
data.loc[
    :2, ["Phi", "Temperature", "Temperature Symbol", "Pressure", "Pressure Symbol"]
]

# separate tuples into two columns

names = ["Phi0", "Phi1"]
for index, column in enumerate(names):
    data[column] = data["Phi"].apply(lambda Phi: Phi[index])
names = ["T0", "T1"]
for index, column in enumerate(names):
    data[column] = data["Temperature"].apply(lambda Temperature: Temperature[index])
names = ["P0", "P1"]
for index, column in enumerate(names):
    data[column] = data["Pressure"].apply(lambda Pressure: Pressure[index])
data.loc[:2, ["Phi", "Phi0", "Phi1", "Temperature", "T0", "T1", "Pressure", "P0", "P1"]]

# select columns to run in algorythm

# comment the columns that u dont want to be in hdbscan
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
    "shift",
]

# hdbscan example

clusterer = hdbscan.HDBSCAN()
clusterer = clusterer.fit(data.loc[:, columns])

# number of clusters found
clusterer.labels_.max()

# probabilities of membership of each point to relative cluster found
pd.Series(clusterer.probabilities_)


# metrics available to run hdbscan
pd.Series(hdbscan.dist_metrics.METRIC_MAPPING.keys())


# create a function to evaluate clusters found and returns the best version of hdbscan


def hdbscanned(input_data, iterations=1):
    best_min_cluster = {
        "n_cluster": 0,
        "probability": 0,
        "n_noise": input_data.shape[0],
        "best_metrics": "euclidean",
    }
    for iteration in range(iterations):
        for metrics in hdbscan.dist_metrics.METRIC_MAPPING.keys():
            for i in range(2, 100):
                clusterer = hdbscan.HDBSCAN(min_cluster_size=i)
                clusterer = clusterer.fit(input_data)
                if (
                    clusterer.probabilities_[clusterer.probabilities_ == 0].shape[0]
                    <= best_min_cluster["n_noise"]
                ):
                    if (
                        clusterer.probabilities_[clusterer.probabilities_ > 0].mean()
                        > best_min_cluster["probability"]
                    ):
                        best_min_cluster["n_cluster"] = i
                        best_min_cluster["probability"] = clusterer.probabilities_[
                            clusterer.probabilities_ > 0
                        ].mean()
                        best_min_cluster["n_noise"] = clusterer.probabilities_[
                            clusterer.probabilities_ == 0
                        ].shape[0]
                        if metrics != best_min_cluster["best_metrics"]:
                            best_min_cluster["best_metrics"] = metrics

    print(f"best_min_cluster = {best_min_cluster}\n")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=best_min_cluster["n_cluster"],
        metric=best_min_cluster["best_metrics"],
    )
    return clusterer.fit(input_data), best_min_cluster


# start hsbscanned on data

input_data = data.loc[:, columns]
clusterer, clusterer_eval = hdbscanned(input_data)

data["ClusterID"] = clusterer.labels_
data.head()
data.to_excel("clustered.xlsx")

data = pd.read_excel("clustered.xlsx", index_col=0)
data.head()


def groupanddescribe(
    data,
    groupby_elem=["ClusterID"],
    slc=["Phi0"],
    statistics={"mean": pd.core.groupby.GroupBy.mean},
    count=False,
):
    # init empty dataframe
    retval = pd.DataFrame()
    if count:
        statistics["count"] = pd.core.groupby.GroupBy.count

    for stat_name, stat in statistics.items():

        grouped = data.groupby(groupby_elem)[slc]
        grouped = stat(grouped)
        m_index = pd.MultiIndex.from_tuples([(x, stat_name) for x in grouped.keys()])
        grouped.columns = m_index
        retval = pd.concat([retval, grouped], axis=1)

    if "count" in list(zip(*retval.keys()))[1]:
        # drop all column with count except the first one
        count_columns = [i for i in retval.keys() if i[1] == "count"]
        retval = retval.drop(columns=count_columns[1:])
        renamed_index = list(retval.keys())
        renamed_index[-1] = list(renamed_index[-1])
        renamed_index[-1][0] = ""
        retval.columns = pd.MultiIndex.from_tuples(renamed_index)
    retval.columns = retval.columns.swaplevel()  # .map('_'.join)
    return retval


stat_names = ["mean", "min", "max", "median", "std", "var"]
stats = {
    stat_name: getattr(pd.core.groupby.GroupBy, stat_name) for stat_name in stat_names
}
metadata = groupanddescribe(data, slc=columns, statistics=stats, count=True)
metadata.head()

metadata = data.groupby(["ClusterID"])[columns]
print(type(metadata))
metadata = pd.core.groupby.GroupBy.mean(metadata)
m_index = pd.MultiIndex.from_tuples([(x, "mean") for x in metadata.keys()])
metadata.columns = m_index
metadata

metadata = data.groupby(["ClusterID"])[columns].describe()
metadata.head()
print(metadata.keys())
# metadata = data.groupby(["ClusterID"])[columns].mean().reset_index()
# metadata.head()

metadata = metadata.reset_index()

pd.set_option("display.max_columns", None)

pd.reset_option("max_columns")

metadata.rename(columns={"Score": "MeanOldScores"}, inplace=True)
metadata["meanScore"] = metadata.loc[:, ["d0L2", "d1L2", "d0Pe", "d1Pe"]].mean(axis=1)
metadata.head()

metadata.to_excel("metadata.xlsx")

metadata.to_csv("metadata.csv")
