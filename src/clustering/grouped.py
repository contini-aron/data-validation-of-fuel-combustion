import os
from . import ClusteringAlgorythm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mkdir(dir_name: str) -> None:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    return


def compute_grouped(
    df: pd.DataFrame,
    columns: list[str],
    groupby: list[str],
    clustering_algo: ClusteringAlgorythm,
    stats=None,
    ignore_val=20,
) -> None:
    """
    :param df: pd.Dataframe of input data
    :param columns: columns to perform cluster on
    :param groupby: column to perform groupby on
    :param clustering_algo: the clusterig algorythm to use
    :param stats: the statistics to be shown
    :return: None
    print("current dir is ", os.path.abspath(os.curdir))
    print("wanted path is ", os.path.abspath(os.path.join(os.curdir, "metadata"+os.sep+"grouped"+ os.sep+f"GroupedBy_{groupby}")))
    group_dir_name = os.path.abspath(os.path.join(os.curdir, "metadata"+os.sep+"grouped"+ os.sep+f"GroupedBy_{groupby}"))
    mkdir(group_dir_name)
    ignored = []
    clustered_groups = []
    for group in df.groupby(groupby):
        # group is a tuple(groupbyID, dataframe grouped)
        data = group[1]
        group_name = group[0]
        if data.shape[0] > ignore_val:
            print(f"GROUP:\t{group_name}")
            data = Clusterer(clustering_algo).start(df, columns)
            clustered_groups.append((group_name, data))
            group_dir = group_dir_name + f"/{group_name}"
            mkdir(group_dir)
            data.to_excel(group_dir + f"/Clustered:{group_name}.xlsx")
            groupanddescribe(data, slc=columns, statistics=stats, count=True).to_excel(
                group_dir + f"/Metadata:{group_name}.xlsx"
            )
            graph_the_data_by_cluster(
                data,
                group_dir,
                ["Temperature0", "Pressure0", "Phi0"],
                True,
                str(group_name) + f"{['Temperature0', 'Pressure0', 'Phi0']}",
            )
            graph_the_data_by_cluster(
                data,
                group_dir,
                ["Temperature1", "Pressure1", "Phi1"],
                True,
                str(group_name) + f"{['Temperature1', 'Pressure1', 'Phi1']}",
            )

        else:
            ignored.append(group_name)
    print(f"ignored values\n{ignored}")
    
    """


# performs descriptive analysis on clustered data
def groupanddescribe(
    data:pd.DataFrame,  # data to be analyzed
    groupby_elem:list[str]=None,  # column to group by
    slc:list[str]=None,  # columns to be analyzed
    statistics:dict=None,  # dictionary of statistics to be shown
    count:bool=False, # if true, counts are shown
):
    """
    creates a multiindex dataframe grouping data by groupby_elem with columns as the stats chosen
    """
    # adds default values to slc(selection), groupby_elem and statistics if not provided
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

    # for every stat in statistics, computes the stat on the data
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
    data,
    directory,
    columns=["Temperature0", "Pressure0", "Phi0"],
    ignore_noise=True,
    title=None,
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
    fig.savefig(f"{directory}/{title}{columns}.png", bbox_inches="tight")
    plt.close()