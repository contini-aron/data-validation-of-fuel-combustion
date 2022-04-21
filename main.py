import hdbscan
import pandas as pd
import ast
import numpy as np
import sys
import argparse

def parse_input(data):
    
    #rename dataframes and apply correct unit measurement
    data.rename(columns = {"Pressure (Bar)":"Pressure", "Temperature (K)":"Temperature"}, inplace=True)
    data["Phi"]=data["Phi"].apply(ast.literal_eval)
    data["Temperature"]=data["Temperature"].apply(ast.literal_eval)
    data["Temperature Symbol"] = "K"
    data["Pressure"]=data["Pressure"].apply(ast.literal_eval)
    data["Pressure Symbol"] = "Bar"
    
    #split tuples into columns
    names=["Phi0", "Phi1"]
    for index,column in enumerate(names):
        data[column] = data['Phi'].apply(lambda Phi: Phi[index])
    names=["T0", "T1"]
    for index,column in enumerate(names):
        data[column] = data['Temperature'].apply(lambda Temperature: Temperature[index])
    names=["P0", "P1"]
    for index,column in enumerate(names):
        data[column] = data['Pressure'].apply(lambda Pressure: Pressure[index])
    
    return data


# performs descriptive analysis on clustered data 
def groupanddescribe(data, groupby_elem = ["ClusterID"],  slc = ["Phi0"], statistics = {"mean":pd.core.groupby.GroupBy.mean}, count=False):
    #init empty dataframe
    retval = pd.DataFrame()
    if count:
        statistics["count"]=pd.core.groupby.GroupBy.count
    
    for stat_name, stat in statistics.items():

        grouped = data.groupby(groupby_elem)[slc]
        grouped = stat(grouped)
        m_index = pd.MultiIndex.from_tuples([(x,stat_name) for x in grouped.keys()])
        grouped.columns=m_index
        retval = pd.concat([retval, grouped], axis = 1)
        
    if "count" in list(zip(*retval.keys()))[1]:
        #drop all column with count except the first one
        count_columns = [i for i in retval.keys() if i[1]=="count"]
        retval = retval.drop(columns = count_columns[1:])
        renamed_index = list(retval.keys())
        renamed_index[-1]=list(renamed_index[-1])
        renamed_index[-1][0]=""
        retval.columns = pd.MultiIndex.from_tuples(renamed_index)

    retval.columns = retval.columns.swaplevel()#.map('_'.join)    
    return retval


parser = argparse.ArgumentParser(description="Cluster Data for hdbscan, can use groupby")
parser.add_argument("-f", "--file", type=str, required=True, help="path to file of input data, you can omit the path if u have the file in the same folder of the script" )
parser.add_argument("-ig", "--input-groupby",default=None, type=str, help="specify if you want to groupby the data before running hdbscan Eg '-ig ['col_name1', 'col_name2']'")
parser.add_argument("-ic", "--input-columns",default=None, type=str, help="specify if you want to use only certain columns of the data into hdbscan Eg '-ic ['col_name1', 'col_name2']'")

if __name__ == "__main__":

    arguments = parser.parse_args()    
    print(arguments)

    yes=False
    if yes:
        data = pd.read_excel("data.xlsx", index_col=0)
        data=parse_input(data)
#data.head()

#comment the columns that u dont want to be in hdbscan
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


        data = pd.read_excel("clustered.xlsx", index_col=0)
# change the stats you would like to see in stat_name if they are part of pandas.core.groupby.GroupBy.*
        stat_names = ["mean", "min", "max", "median", "std", "var"]
        stats = {stat_name:getattr(pd.core.groupby.GroupBy, stat_name) for stat_name in stat_names}
        metadata = groupanddescribe(data, slc=columns, statistics = stats, count=True)
        metadata.to_excel("metadata.xlsx")
        metadata.to_csv("metadata.csv")

# set option to print all the columns
        pd.set_option('display.max_columns', None)
        metadata.head()

