"""
boy
"""
import os
import pandas as pd
from clustering import Clusterer
from clustering import input_parse as ip
from clustering import AlgoHDBSCAN
from clustering import AlgoAffinityPropagation
from clustering import AlgoKmeans
from clustering import plot_percentiles
from clustering import get_columns, get_groupby
from clustering import compute_grouped, groupanddescribe, graph_the_data_by_cluster


if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.join(os.path.curdir, "src")))
    print("cwd is" + os.path.abspath(os.curdir))
    cluster = True
    columns = get_columns()
    # change the stats you would like to see in stat_name if they are part of pandas.core.groupby.GroupBy.*
    stat_names = ["mean", "min", "max", "median", "std", "var"]
    stats = {
        stat_name: getattr(pd.core.groupby.GroupBy, stat_name)
        for stat_name in stat_names
    }
    if cluster:
        file = os.path.join(os.path.join(os.path.curdir, "input_files"), "data.xlsx")
        print(file)
        print(os.path.abspath(file))
        df = pd.read_excel("./input_files/data.xlsx")
        df = ip.parse(df)

        # clusterer = Clusterer(AlgoAffinityPropagation)
        # print(clusterer.start(df, columns))
        #clusterer = Clusterer(AlgoKmeans)
        #data = clusterer.start(df, columns)
        #print(clusterer.clustered)
        #metadata = clusterer.get_metadata(slc=columns, statistics_names=stat_names, count=True)
        #print(clusterer.metadata)
        #clusterer.get_best_clusters()
        #clusterer.parallel_coordinates_plot()
        #data = clusterer.start(df, columns, groupby=["Target"])
        #clusterer.export(groupby=True)



        clusterer = Clusterer(AlgoHDBSCAN)
        #data = clusterer.start(df, columns)
        data = pd.read_excel(f"{os.curdir}{os.sep}metadata{os.sep}clustered.xlsx")
        clusterer.get_metadata(data=data, slc=columns, statistics_names=stat_names, count=True)
        clusterer.get_best_clusters()
        clusterer.parallel_coordinates_plot()
        print(data)
        print(data["ClusterID"].max())
        plot_percentiles(df.loc[:, get_columns()])
        data.to_excel("./metadata/clustered.xlsx")

    data = pd.read_excel("./metadata/clustered.xlsx")
    print("bro type is :", type(stats["mean"]))
    metadata = groupanddescribe(data, slc=columns, statistics=stats, count=True)
    metadata.to_excel("./metadata/metadata.xlsx")
    print(metadata)
    graph_the_data_by_cluster(
        data,
        "./metadata/whole_dataset",
        ["Temperature1", "Pressure1", "Phi1"],
        ignore_noise=True,
        title="WHOLE DATASET",
    )
    graph_the_data_by_cluster(
        data,
        "./metadata/whole_dataset",
        ["Temperature0", "Pressure0", "Phi0"],
        ignore_noise=True,
        title="WHOLE DATASET",
    )

    graph_the_data_by_cluster(
        data,
        "./metadata/whole_dataset",
        ["Score", "d0L2", "d1L2"],
        ignore_noise=True,
        title="WHOLE DATASET",
    )
    df = pd.read_excel("./input_files/data.xlsx")
    df = ip.parse(df)
    compute_grouped(df, get_columns(), get_groupby(), AlgoKmeans, stats)