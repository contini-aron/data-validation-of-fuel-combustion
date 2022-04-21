
import pandas as pd
import scipy as spy
import matplotlib.pyplot as plt
import ast
import sklearn.metrics as skt
from pyclustering.cluster.cure import cure
from pyclustering.utils import read_sample
from pyclustering.cluster import cluster_visualizer_multidim

# read input excel 

data = pd.read_excel("data.xlsx")

# rename indexes

data.rename(columns = {"Unnamed: 0":"main_index", "Pressure (Bar)":"Pressure", "Temperature (K)":"Temperature"}, inplace=True)
data.head()

# transform Phi, Temperature(K) column into a column of tuples

data["Phi"]=data["Phi"].apply(ast.literal_eval)
data["Temperature"]=data["Temperature"].apply(ast.literal_eval)
data["Pressure"]=data["Pressure"].apply(ast.literal_eval)
data.loc[:2,["Phi","Temperature", "Pressure"]]

# separate tuples into two columns 

names=["Phi0", "Phi1"]
for index,column in enumerate(names):
    data[column] = data['Phi'].apply(lambda Phi: Phi[index])
names=["T0", "T1"]
for index,column in enumerate(names):
    data[column] = data['Temperature'].apply(lambda Temperature: Temperature[index])
names=["P0", "P1"]
for index,column in enumerate(names):
    data[column] = data['Pressure'].apply(lambda Pressure: Pressure[index])
data.loc[:2,["Phi","Phi0","Phi1", "Temperature", "T0", "T1", "Pressure", "P0", "P1"]] 

# #grouping the dataframe
# group_names = ['Experiment Type', 'Reactor', 'Target', 'Fuels', 'Phi', 'Pressure (Bar)', 'Temperature (K)', 'd0L2', 'd1L2', 'd0Pe', 'd1Pe', 'shift', 'Phi0', 'Phi1']
# grouped = data.groupby(group_names)
# groups = [x.reset_index() for _, x in grouped]
# for i in groups:
#     print(i.loc[:,["main_index", "Experiment Type", "Target", "Fuels"]])
#     print(type(i),"\n\n")

# Example of pyclustering library

# choose the variable to consider into clustering algorythm
# column_names = ['Phi0', 'Phi1', 'P0', 'P1', 'T0', 'T1', 'd0L2', 'd1L2', 'd0Pe', 'd1Pe', 'shift']
column_names = ['Phi0', 'Phi1', 'P0', 'P1', 'T0', 'T1']
data.loc[:,column_names]
input_data = list(data.loc[:,column_names].to_numpy())
# Allocate clusters.
cure_instance = cure(input_data, 4, number_represent_points=1);
cure_instance.process();
clusters = cure_instance.get_clusters();
cluster_encoding = cure_instance.get_cluster_encoding()
means = cure_instance.get_means()
representors = cure_instance.get_representors()
print(representors)

# function to unpack clusters to dataframe
def assign_cluster_to_dataframe(clusters, df, col_name):
    df[col_name]=0
    for cluster_num, cluster in enumerate(clusters):
        for data_row in cluster: 
            df.at[data_row, col_name] = cluster_num+1
    return df

data = assign_cluster_to_dataframe(clusters, data, "ClusterID")    
print(data.loc[:20, ["ClusterID"]])
skt.silhouette_score(clusters, input_data)
# Visualize allocated clusters.
# visualizer = cluster_visualizer_multidim()
# visualizer.append_clusters(clusters, input_data)
# visualizer.show()

