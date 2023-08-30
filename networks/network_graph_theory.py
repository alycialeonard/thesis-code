# Graph theory process for last chapter of thesis and Katiri case study
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk
# Date: 2022-02-06

import time
from datetime import datetime
import pandas as pd
import geopandas as gp
import networkx as nx
import sklearn.cluster as cluster
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import Delaunay
from scipy.spatial import distance
from kneed import KneeLocator
import sys
from math import radians, cos, sin, asin, sqrt
import osmnx as ox
from shapely.geometry import Point

# --------------- FILEPATHS TO DEFINE --------------- #

# Katiri building centroids
data = 'D:\\clustering\\katiri-centroids.geojson'

# Places to save results
clustering_results = 'D:\\clustering\\results\\clustering_results.geojson'
clustering_results_no_outliers = 'D:\\clustering\\results\\clustering_results_no_outliers.geojson'
analysis_results_csv = 'D:\\clustering\\results\\analysis_results.csv'
analysis_results = 'D:\\clustering\\results\\analysis_results.geojson'
console_output = 'D:\\clustering\\results\\output.txt'
knee_graph = 'D:\\clustering\\results\\knee_graph.png'
cluster_graph = 'D:\\clustering\\results\\cluster_graph.png'
tree_shapefiles = 'D:\\clustering\\results\\shapefile_exports'


# --------------- PRE-PROCESSING --------------- #

# Make sure we print all columns during df printing
pd.set_option('display.max_columns', 500)

# Get process start time
start_time = datetime.now()

# Uncomment the following to set stdout to print to output file
# sys.stdout = open(console_output, 'w')

# Import cluster data as GeoDataFrame
print('Reading cluster data into a GeoDataFrame...')
gdf = gp.read_file(clustering_results)
print('GeoDataFrame is ready!')

# Get unique cluster label values and number (N) of unique clusters - includes outlier cluster (-1)
cluster_ids = gdf['cluster_index_1'].unique()
N = len(cluster_ids)

# --------------- DISTRIBUTION GRIDS --------------- #

print('Generating distribution grid for each cluster...')
#
# Instantiate dictionary to hold mean coords, no. of buildings, Delaunay tri, and min span tree for each cluster
mean_coord = {}
building_count = {}
delaunay = {}
min_span_tree = {}
total_conductor_length_mst = {}
total_conductor_length_radial = {}
locations = {}

# For each cluster, get mean coordinate, number of buildings, and generate network estimate
for cluster in cluster_ids:

    # Skip cluster_id = -1 (cluster of outliers, don't need stats or network)
    if cluster == -1:
        continue

    # Get a subset of gdf for the cluster being considered
    subset = gp.GeoDataFrame(gdf.loc[gdf['cluster_index_1'] == cluster])

    # Get mean point of each cluster
    mean_coord[cluster] = [subset['x'].mean(), subset['y'].mean()]

    # Get and record the number of points in that cluster
    no_points_cluster = subset.shape[0]
    building_count[cluster] = no_points_cluster

    # Get x and y columns from the GeoDataFrame and convert into a numpy array
    subset_coords = subset.drop(['cluster_index_1', 'geometry'], axis=1).to_numpy()

    # Get and record Delauney triangulation of subset coordinates
    tri = Delaunay(subset_coords)
    delaunay[cluster] = tri

    # Get neighbouring vertices of vertices
    indices = tri.vertex_neighbor_vertices[0]
    indptr = tri.vertex_neighbor_vertices[1]

    # Instantiate dictionary to hold neighbors of each point & data-frame to hold distances between neighbours
    neighbors = {}
    loc = {}
    distances = pd.DataFrame(columns=["source", "dest", "distance"])

    # Get dictionary of neighbors of all points in cluster and a dictionary of locations of all points in cluster
    for k in range(0, no_points_cluster):
        neighbors[k] = indptr[indices[k]:indices[k + 1]]
        loc[k] = subset_coords[k][0], subset_coords[k][1]
    locations[cluster] = loc

    # Get distances between all Delaunay neighbors
    for key, values in neighbors.items():
        for value in values:
            coord_1 = subset_coords[key]
            coord_2 = subset_coords[value]
            dist = distance.euclidean(coord_1, coord_2)
            distances = distances.append({"source": key, "dest": value, "distance": dist}, ignore_index=True)

    # Create a graph from this information
    G = nx.Graph()
    for index, row in distances.iterrows():
        G.add_edge(row['source'], row['dest'], weight=row['distance'])

    # Make and record a minimum spanning tree of graph
    T = nx.minimum_spanning_tree(G)
    min_span_tree[cluster] = T

    # Get and record total distance of all edges in the tree (i.e. conductor in dist grid)
    edges = T.edges(data=True)
    weights = [x[2]['weight'] for x in edges]
    total_dist = sum(weights)
    total_conductor_length_mst[cluster] = total_dist

    # Save minimum spanning tree as shapefile
    T_save = nx.relabel_nodes(T, loc)
    T_dir = T_save.to_directed()
    #nx.write_shp(T_dir, tree_shapefiles + '\\cluster' + str(cluster))

    d = 0
    # Get distance of all points to mean coordinate
    for i in range(0, no_points_cluster):
        coord_1 = subset_coords[i]
        coord_2 = mean_coord[cluster]
        dist = distance.euclidean(coord_1, coord_2)
        d = d + dist
    total_conductor_length_radial[cluster] = d

print("Lengths of conductor used in each community MST cluster: ", total_conductor_length_mst)
print("Lengths of conductor used in each community radial cluster: ", total_conductor_length_radial)

# --------------- TRANSMISSION GRID --------------- #

print('Generating transmission grid between clusters...')

# Create dataframe where rows are clusters, columns are cluster mean coordinates
mean_coord_df = pd.DataFrame.from_dict(mean_coord).transpose()
# Get the number of clusters
no_clusters = mean_coord_df.shape[0]
# Convert coordinates to numpy array
mean_coord_array = mean_coord_df.to_numpy()

# Get Delaunay triangulation of mean coordinates
tri = Delaunay(mean_coord_array)
# Get neighbouring vertices of vertices
indices = tri.vertex_neighbor_vertices[0]
indptr = tri.vertex_neighbor_vertices[1]

# Instantiate dictionary to hold neighbors of each point & data-frame to hold distances between neighbours
neighbors = {}
loc = {}
distances = pd.DataFrame(columns=["source", "dest", "distance"])

# Get dictionary of neighbors of all points in cluster and a dictionary of locations of all points in cluster
for k in range(0, no_clusters):
    neighbors[k] = indptr[indices[k]:indices[k + 1]]
    loc[k] = mean_coord_array[k][0], mean_coord_array[k][1]

# Get distances between all Delaunay neighbors
for key, values in neighbors.items():
    for value in values:
        coord_1 = mean_coord_array[key]
        coord_2 = mean_coord_array[value]
        dist = distance.euclidean(coord_1, coord_2)
        distances = distances.append({"source": key, "dest": value, "distance": dist}, ignore_index=True)

# Create a graph from this information
G = nx.Graph()
for index, row in distances.iterrows():
    G.add_edge(row['source'], row['dest'], weight=row['distance'])

# Make a minimum spanning tree of graph
T = nx.minimum_spanning_tree(G)

# Get total distance of all edges in the tree (i.e. conductor in dist grid)
edges = T.edges(data=True)
weights = [x[2]['weight'] for x in edges]
total_transmission_dist = sum(weights)

print("Total length of transmission line used: ", total_transmission_dist)

# Save transmission grid
T_save = nx.relabel_nodes(T, loc)
T_dir = T_save.to_directed()
#nx.write_shp(T_dir, tree_shapefiles + '/transmission')

# Get graph of roads and centroids

# Import road line data as GeoDataFrame
gdf_edges = gp.read_file('D:\\clustering\\katiri-osm-roads.geojson')
# Import road point data as GeoDataFrame
gdf_nodes = gp.read_file('D:\\clustering\\osm_roads_points\\katiri_osm_roads_points.shp')
gdf_nodes['x'] = gdf_nodes.geometry.x
gdf_nodes['y'] = gdf_nodes.geometry.y
#print(gdf_nodes.head())
#G = ox.graph_from_gdfs(gdf_nodes, gdf_edges)

# Get GDF of cluster centroids
gdf_centroids = gp.GeoDataFrame(mean_coord_df, geometry=gp.points_from_xy(mean_coord_df.iloc[:, 0], mean_coord_df.iloc[:, 1]))
gdf_centroids.rename(columns={0: 'x', 1: 'y'}, inplace=True)
# gdf_centroids.to_file('D:\\clustering\\results\\cluster_centroids_eps77.geojson', driver='GeoJSON')


# gdf_centroids = gdf_centroids.set_crs(crs="EPSG:3857")
# gdf_centroids["x"] = gdf_centroids.geometry.x
# gdf_centroids["y"] = gdf_centroids.geometry.y
#gdf_centroids["x"] = pd.to_numeric(gdf_centroids["x"], downcast="float")
#gdf_centroids["y"] = pd.to_numeric(gdf_centroids["y"], downcast="float")
# print(gdf_centroids.head())
#G_roads = ox.graph_from_gdfs(gdf_centroids, gdf_roads)
#G_roads = ox.graph_from_address("Katiri, Sierra Leone")
#route = nx.shortest_path(G, node1, node2, weight='length') #obtain shortest path based on length
#route_length = nx.shortest_path_length(G, node1, node2, weight='length') #obtain shortest path length
# --------------- PLOTS --------------- #

# Plot Minimum Spanning Trees made from Delaunay Triangulation
plt.figure()
for key in min_span_tree.keys():
    if key == -1:
        plt.scatter(x=gdf.loc[gdf['cluster_index_1'] == key, 'x'],
                    y=gdf.loc[gdf['cluster_index_1'] == key, 'y'],
                    s=20, color='black', alpha=1)
        continue
    nx.draw_networkx(min_span_tree[key], pos=locations[key], with_labels=False, node_size=15)
nx.draw_networkx(T, pos=loc, with_labels=False, node_size=15, node_color='r')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
axes = plt.gca()

plt.show(block=False)

