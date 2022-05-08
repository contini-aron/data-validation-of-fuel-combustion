"""
boy
"""
import os
import pandas as pd
import input_parse as ip
from src.clusterer import Clusterer
from clustering_algo.algo_hdbscan import AlgoHDBSCAN
from clustering_algo.algo_affinitypropagation import AlgoAffinityPropagation
from clustering_algo.algo_kmeans import AlgoKmeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.clustering_algo.clustering_algorythm import ClusteringAlgorythm


def get_groupby():
    return [
        # "Experiment DOI",
        # "Exp SciExpeM ID",
        # "Chem Model",
        # "Chem Model ID",
        # "Experiment Type",
        # "Reactor",
        "Target",
        # "Fuels",
        # "Phi",
        # "Phi0",
        # "Phi1",
        # "Pressure",
        # "P0",
        # "P1",
        # "Temperature",
        # "T0",
        # "T1",
        # "Score",
        # "Error",
        # "d0L2",
        # "d1L2",
        # "d0Pe",
        # "d1Pe",
        # "shift"
    ]


def get_columns() -> list[str]:
    """
    :return: default column names used for clustering
    """
    return [
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
        "Pressure0",
        "Pressure1",
        # "Temperature",
        "Temperature0",
        "Temperature1",
        "Score",
        "Error",
        "d0L2",
        "d1L2",
        "d0Pe",
        "d1Pe",
        "shift",
    ]

def mkdir(dir_name: os.path)->None:
    if not os.isdir(dir_name):
        os.mkdir(dir_name)
    return
def compute_grouped(
    df: pd.DataFrame,
    columns: list[str],
    groupby: list[str],
    clustering_algo: ClusteringAlgorythm,
    stats: list[str] = None,
) -> None:
    """
    :param df: pd.Dataframe of input data
    :param columns: columns to perform cluster on
    :param groupby: column to perform groupby on
    :param clustering_algo: the clusterig algorythm to use
    :param stats: the statistics to be shown
    :return: None
    """
    group_dir_name = f"./metadata/grouped/GroupedBy:{groupby}"
    mkdir(group_dir_name)
    ignored = []
    clustered_groups = []
    for group in df.groupby(groupby):
        # group is a tuple(groupbyID, dataframe grouped)
        data = group[1]
        group_name = group[0]
        if data.shape[0] > 20:
            print(f"GROUP:\t{group_name}")
            data = Clusterer(clustering_algo).start(df, columns)
            clustered_groups.append((group_name, data))

        else:
            ignored.append(group_name)

    print(f"ignored values\n{ignored}")
    groupanddescribe(clustered_groups[0][1], slc=columns, statistics=stats, count=True)


# performs descriptive analysis on clustered data
def groupanddescribe(
    data,
    groupby_elem=None,
    slc=None,
    statistics=None,
    count=False,
):
    if slc is None:
        slc = ["Phi0"]
    if groupby_elem is None:
        groupby_elem = ["ClusterID"]
    if statistics is None:
        statistics = {"mean": pd.core.groupby.GroupBy.mean}
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


def graph_the_data_by_cluster(
    data, columns=["Temperature0", "Pressure0", "Phi0"], ignore_noise=False, title=None
):
    fig = plt.figure(figsize=(12, 9))
    ax = Axes3D(fig, auto_add_to_figure=False)

    if ignore_noise:
        data = data.loc[data["ClusterID"] != -1, :]

    grouped = data.groupby("ClusterID").groups.items()
    for grp_name, grp_idx in grouped:
        y_axis = data.loc[grp_idx, columns[0]]
        x_axis = data.loc[grp_idx, columns[1]]
        z_axis = data.loc[grp_idx, columns[2]]
        ax.scatter(
            x_axis,
            y_axis,
            z_axis,
            label="ClusterID = " + str(grp_name),
            cmap="coolwarm",
        )
    ax.set_xlabel("\n" + columns[0], linespacing=4)
    ax.set_ylabel("\n" + columns[1], linespacing=4)
    ax.set_zlabel("\n" + columns[2], linespacing=4)

    ax.legend(ncol=4, bbox_to_anchor=(2, 1), loc="upper right", title=title)
    fig.add_axes(ax)
    print(os.curdir)
    fig.savefig(f"./metadata/whole_dataset/{title}{columns}.png", bbox_inches="tight")


if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.join(os.path.curdir, "src")))
    print("cwd is" + os.path.abspath(os.curdir))
    cluster = False
    columns = get_columns()
    if cluster:
        file = os.path.join(os.path.join(os.path.curdir, "input_files"), "data.xlsx")
        print(file)
        print(os.path.abspath(file))
        df = pd.read_excel("./input_files/data.xlsx")
        df = ip.parse(df)

        clusterer = Clusterer(AlgoAffinityPropagation)
        print(clusterer.start(df, columns))
        clusterer = Clusterer(AlgoKmeans)
        print(clusterer.start(df, columns))
        clusterer = Clusterer(AlgoHDBSCAN)
        data = clusterer.start(df, columns)
        print(data)
        print(data["ClusterID"].max())
        data.to_excel("./metadata/clustered.xlsx")

    data = pd.read_excel("./metadata/clustered.xlsx")
    # change the stats you would like to see in stat_name if they are part of pandas.core.groupby.GroupBy.*
    stat_names = ["mean", "min", "max", "median", "std", "var"]
    stats = {
        stat_name: getattr(pd.core.groupby.GroupBy, stat_name)
        for stat_name in stat_names
    }
    metadata = groupanddescribe(data, slc=columns, statistics=stats, count=True)
    metadata.to_excel("./metadata/metadata.xlsx")
    print(metadata)
    graph_the_data_by_cluster(
        data,
        ["Temperature1", "Pressure1", "Phi1"],
        ignore_noise=True,
        title="WHOLE DATASET",
    )
    graph_the_data_by_cluster(
        data,
        ["Temperature0", "Pressure0", "Phi0"],
        ignore_noise=True,
        title="WHOLE DATASET",
    )
