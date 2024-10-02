import os
import csv
import pickle
import pdb
import numpy as np
import pandas as pd 
from geopy.distance import geodesic

def get_adjacency_matrix(distance_df_filename: str, num_of_vertices: int, id_filename: str = None) -> tuple:
    """Generate adjacency matrix.

    Args:
        distance_df_filename (str): path of the csv file contains edges information
        num_of_vertices (int): number of vertices
        id_filename (str, optional): id filename. Defaults to None.

    Returns:
        tuple: two adjacency matrix.
            np.array: connectivity-based adjacency matrix A (A[i, j]=0 or A[i, j]=1)
            np.array: distance-based adjacency matrix A
    """

    adj_mx = pd.read_csv(distance_df_filename, index_col=0)
    adj_mx = adj_mx[adj_mx.iloc[:,1].isin(id_filename)]

    adjacency_matrix_city = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    adjacency_matrix_river = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    adjacency_matrix_distance = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    identity = np.identity(adj_mx.shape[0])

    for i in range(num_of_vertices):
        for j in range(num_of_vertices):
            if adj_mx.iloc[i,3] == adj_mx.iloc[j,3]:
                adjacency_matrix_city[i, j] = 1

            if adj_mx.iloc[i,5] == adj_mx.iloc[j,5]:
                adjacency_matrix_river[i, j] = 1

            point1 = (np.float(adj_mx.iloc[i,9][:-1]), np.float(adj_mx.iloc[i,8][:-1]))  
            point2 = (np.float(adj_mx.iloc[j,9][:-1]), np.float(adj_mx.iloc[j,8][:-1])) 
          
            # 计算距离
            adjacency_matrix_distance[i,j] = geodesic(point1, point2).kilometers

    # exp(-dist**2 / sigma**2)
    adjacency_matrix_distance = np.exp(-adjacency_matrix_distance**2 / adjacency_matrix_distance.var())
    
    # 统一不保留自环
    adjacency_matrix_distance = adjacency_matrix_distance - identity
    adjacency_matrix_city = adjacency_matrix_city - identity
    adjacency_matrix_river = adjacency_matrix_river - identity
    return adjacency_matrix_city, adjacency_matrix_river, adjacency_matrix_distance
    
        
       

def get_directed_adjacency_matrix(distance_df_filename: str, num_of_vertices: int, id_filename: str = None) -> tuple:
    """Generate adjacency matrix.

    Args:
        distance_df_filename (str): path of the csv file contains edges information
        num_of_vertices (int): number of vertices
        id_filename (str, optional): id filename. Defaults to None.

    Returns:
        tuple: two adjacency matrix.
            np.array: connectivity-based adjacency matrix A (A[i, j]=0 or A[i, j]=1)
            np.array: distance-based adjacency matrix A
    """

    if "npy" in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        adjacency_matrix_connectivity = np.zeros((int(num_of_vertices), int(
            num_of_vertices)), dtype=np.float32)
        adjacency_matrix_distance = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                                             dtype=np.float32)
        if id_filename:
            # the id in the distance file does not start from 0, so it needs to be remapped
            with open(id_filename, "r") as f:
                id_dict = {int(i): idx for idx, i in enumerate(
                    f.read().strip().split("\n"))}  # map node idx to 0-based index (start from 0)
            with open(distance_df_filename, "r") as f:
                f.readline()  # omit the first line
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    adjacency_matrix_connectivity[id_dict[i], id_dict[j]] = 1
                    adjacency_matrix_distance[id_dict[i], id_dict[j]] = distance
            return adjacency_matrix_connectivity, adjacency_matrix_distance
        else:
            # ids in distance file start from 0
            with open(distance_df_filename, "r") as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    adjacency_matrix_connectivity[i, j] = 1
                    adjacency_matrix_distance[i, j] = distance
            return adjacency_matrix_connectivity, adjacency_matrix_distance


def generate_adj_wp(selected_stations, directed=False):

    rainfall_df_filename, num_of_vertices = "datasets/raw_data/{0}/{0}.npz".format("WaterQuality"), 74
    stations_df_filename = "datasets/raw_data/{0}/{1}.csv".format("WaterQuality", "station")

    # if directed:
    #     adj_mx, distance_mx = get_directed_adjacency_matrix(rainfall_df_filename, num_of_vertices, id_filename=id_filename)
    # adj_mx = [adj_mx_city, adj_mx_river, distance_mx]
    adj_mx = get_adjacency_matrix(stations_df_filename, num_of_vertices, id_filename=selected_stations)
    
    # the self loop is missing
    add_self_loop = False
    if add_self_loop:
        print("adding self loop to adjacency matrices.")
        adj_mx_city = adj_mx_city + np.identity(adj_mx_city.shape[0])
        adj_mx_river = adj_mx_river + np.identity(adj_mx_river.shape[0])
        distance_mx = distance_mx + np.identity(distance_mx.shape[0])
    else:
        print("kindly note that there is no self loop in adjacency matrices.")

    with open("datasets/raw_data/WaterQuality/adj_WaterQuality.pkl", "wb") as f:
        pickle.dump(adj_mx, f)
    # with open("datasets/raw_data/WaterQuality/adj_WaterQuality_distance.pkl", "wb") as f:
    #     pickle.dump(distance_mx, f)
