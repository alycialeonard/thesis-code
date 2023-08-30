# Clustering + network design for rural electrification (from PowerAfrica)
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk
# Date: 2021-04-06

import sys
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import geopandas as gp
import networkx as nx
import sklearn.cluster as cluster
from sklearn.neighbors import NearestNeighbors
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# --------------- PARAMETERS --------------- #

# Minimum number of homes in a DBSCAN cluster
DBSCAN_MINPTS = 5

# Number of nearest neighbours to consider in knee-point calculations
K = 5

# Whether legal under-grid distance will be considered, and distance to consider if so
under_grid_considered = False
under_grid = 0.2

# --------------- CONSTANTS --------------- #

R_EARTH_KM = 6371

# --------------- FILEPATHS TO DEFINE --------------- #

# Filepath to OpenStreetMap building footprints data as GeoJSON or ShapeFile, trimmed to region of interest
# This can be downloaded from http://download.geofabrik.de/africa.html as a ShapeFile
buildings = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\gis_osm_buildings_a_free_1_kenema_district.geojson'

# Filepaths to save final results
clustering_results = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\result_convert_revert\\clustering_results.geojson'
analysis_results_csv = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\result_convert_revert\\analysis_results.csv'
analysis_results = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\result_convert_revert\\analysis_results.geojson'
console_output = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\result_convert_revert\\output.txt'
knee_graph = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\result_convert_revert\\knee_graph.png'
cluster_graph = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\result_convert_revert\\cluster_graph.png'
tree_shapefiles = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\result_convert_revert\\shapefile_exports'

# --------------- FUNCTIONS --------------- #

# Find the knee of a set of data following a curve up or down
# Inputs: number of points in dataset, ordered datapoints
# This is based on the Kneedle approach: https://raghavan.usc.edu//papers/kneedle-simplex11.pdf
# Implementation based on the IBM Watson tutorial here:
# https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1
def find_knee(n_points, data):
    # Stack all points and get the first point
    all_coord = np.vstack((range(n_points), data)).T
    first_point = all_coord[0]
    # Get vector between first and last point i.e. the line
    line_vec = all_coord[-1] - all_coord[0]
    # This is a unit vector in the direction of the line
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
    # Find the distance between all points and first point
    vec_from_first = all_coord - first_point
    # Split vecFromFirst into parallel/perpendicular comps. Take the norm of perpendicular comp & get distance.
    # Project vecFromFirst onto line by taking scalar product with unit vector in direction of line
    scalar_product = np.sum(vec_from_first * np.matlib.repmat(line_vec_norm, n_points, 1), axis=1)
    # Multiply scalar product by unit vector
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    # Perpendicular vector to the line is the difference between vecFromFirst and the parallel bit
    vec_to_line = vec_from_first - vec_from_first_parallel
    # Distance to line is the norm of vecToLine
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    # Knee/elbow is the point with the max distance value
    knee_point_index = np.argmax(dist_to_line)
    knee_point_value = data[knee_point_index]
    return knee_point_index, knee_point_value


# Haversine formula for kilometer distance between two lat/long points - https://www.geeksforgeeks.org/program-distance-two-points-earth/
def haversine_dist_from_coords(lat1, lon1, lat2, lon2):
    # The math module contains a function named radians which converts from degrees to radians.
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371
    # calculate and return the result
    return c * r


# --------------- PROCESS --------------- #

# Get process start time
start_time = datetime.now()

# Set stdout to print to output file
# sys.stdout = open(console_output, 'w')

# Import data as GeoDataFrame
print('Reading building footprint data into a GeoDataFrame...')
OSM_data = gp.read_file(buildings)
print('GeoDataFrame is ready!')

# Pre-process data to get lat/long array (X) and number of points (no_points)
print('Pre-processing building footprint data...')
# Get the number of rows (points) in the GeoDataFrame
no_points = OSM_data.shape[0]
print("number of points:", no_points)
# Get centroid points of building polygons
OSM_data['centroid'] = OSM_data['geometry'].centroid
# Break out lat and long into separate columns of GeoDataFrame
OSM_data['lon'] = OSM_data.centroid.x
OSM_data['lat'] = OSM_data.centroid.y
# Get lat and long columns from the GeoDataFrame and convert into a numpy array
coordinates = OSM_data.drop(['name', 'type', 'code', 'fclass', 'osm_id', 'geometry', 'centroid'], axis=1).to_numpy()
print('Array of building centroids is ready for analysis!')

# Convert latitude and longitude points in X from degrees (X) to radians (X_rad)
coordinates_rad = np.radians(coordinates)

# Get n nearest neighbours to each point (fit to points and query on points, nearest neighbour of each is itself)
print('Nearest neighbors calculation on centroids starting...')
nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto', metric='haversine').fit(coordinates_rad)
# Note that both inputs and outputs are radians when using haversine distance metric
distances_rad, indices = nbrs.kneighbors(coordinates_rad)
print('Nearest neighbors complete!')

# Multiply distances by radius of earth in kilometers so the knee value graph is in in kilometer distances
distances_km = distances_rad * R_EARTH_KM
# Sort distances to nth neighbour for all points in ascending order
distance_km = np.asarray(sorted(distances_km[:, K-1]))

# Get the knee point of the ordered distances
print('Calculating knee-point of nearest neighbour distances...')
knee_index, knee_value = find_knee(no_points, distance_km)
print("Knee of the curve is at index = ", knee_index, ", and value = ", knee_value, " kilometers.")

# If under-grid distance is being considered, and the knee value is less than it, reassign it to under-grod
if (knee_value < under_grid) and under_grid_considered:
    knee_value = under_grid
    print('Knee value is less than legal under-grid distance - bumping up to under-grid distance')

# Cluster data using DBSCAN with eps=knee_value in deg
print("Clustering building footprint centroids using DBSCAN (eps = knee point)...")
clustering = cluster.DBSCAN(eps=(knee_value/R_EARTH_KM), min_samples=DBSCAN_MINPTS, algorithm='ball_tree', metric='haversine').fit(coordinates_rad)
print("Clustering complete!")

# Append cluster labels to the gdf of data
OSM_data['cluster_index_1'] = clustering.labels_
# Get number of DBSCAN clusters created (0 to N)
N = np.amax(clustering.labels_)

# Drop centroids column so data can be saved as GeoJSON (only one geometry column allowed)
# Centroid is still preserved in lat and long columns
OSM_data_save = OSM_data.drop(['centroid'], axis=1)

print("Saving clustering results as GeoJSON...")
# Save OSM data with both levels of clustering included as a GeoJSON
OSM_data_save.to_file(clustering_results, driver='GeoJSON')
print("Results saved! Location: ", clustering_results)

clust_exec_time = datetime.now() - start_time
print('Execution time for clustering: ', clust_exec_time)

print('Generating distribution grid for each cluster...')

# Get unique cluster label values
cluster_ids = OSM_data['cluster_index_1'].unique()

# Instantiate dictionary to hold mean coords, no. of buildings, Delaunay tri, and min span tree for each cluster
mean_coord = {}
building_count = {}
delaunay = {}
min_span_tree = {}
total_conductor_length = {}
total_conductor_length_radial = {}
locations = {}

# For each cluster, get mean coordinate, number of buildings, and generate network estimate
for cluster in cluster_ids:

    # NEW: Skip for cluster_id = -1 (cluster of outliers, don't need stats or network)
    if cluster == -1:
        continue

    # Get a subset of gdf for the cluster being considered
    subset = gp.GeoDataFrame(OSM_data.loc[OSM_data['cluster_index_1'] == cluster])

    # Get mean point of each cluster - CHANGE THIS to subset.['lon', 'lat']. mean ???
    mean_coord[cluster] = [subset['lon'].mean(), subset['lat'].mean()]
    # Get number of points in that cluster
    no_points_cluster = subset.shape[0]
    building_count[cluster] = no_points_cluster

    # Get lat and long columns from the GeoDataFrame and convert into a numpy array
    subset_coords = subset.drop(['name', 'type', 'code', 'fclass', 'osm_id', 'geometry', 'centroid', 'cluster_index_1'], axis=1).to_numpy()
    # Get Delauney triangulation of subset coordinates
    tri = Delaunay(subset_coords)
    delaunay[cluster] = tri
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
            dist = haversine_dist_from_coords(coord_1[1], coord_1[0], coord_2[1], coord_2[0])
            distances = distances.append({"source": key, "dest": value, "distance": dist}, ignore_index=True)

    # Create a graph from this information
    G = nx.Graph()
    for index, row in distances.iterrows():
        G.add_edge(row['source'], row['dest'], weight=row['distance'])

    # Make a minimum spanning tree of graph
    T = nx.minimum_spanning_tree(G)
    min_span_tree[cluster] = T

    # Get total distance of all edges in the tree (i.e. conductor in dist grid)
    edges = T.edges(data=True)
    weights = [x[2]['weight'] for x in edges]
    total_dist = sum(weights)
    total_conductor_length[cluster] = total_dist

    # Save minimum spanning tree as shapefile
    T_save = nx.relabel_nodes(T, loc)
    T_dir = T_save.to_directed()
    nx.write_shp(T_dir, tree_shapefiles + '/cluster' + str(cluster))

    d = 0
    # Get distance of all points to mean coordinate
    for i in range(0, no_points_cluster):
        coord_1 = subset_coords[i]
        coord_2 = mean_coord[cluster]
        dist = haversine_dist_from_coords(coord_1[1], coord_1[0], coord_2[1], coord_2[0])
        d = d + dist
    total_conductor_length_radial[cluster] = d

print("Lengths of conductor used in each community MST cluster: ", total_conductor_length)
print("Lengths of conductor used in each community radial cluster: ", total_conductor_length_radial)

# Get "transmission lines" between clusters (i.e. minimum spanning tree of mean coordinates
# Create dataframe where rows are clusters, columns are cluster mean longitude and latitude
mean_coord_df = pd.DataFrame.from_dict(mean_coord).transpose()
# Drop the cluster -1 which is actually outliers
mean_coord_df = mean_coord_df.drop([-1])
no_clusters = mean_coord_df.shape[0]
mean_coord_array = mean_coord_df.to_numpy()

# Get Delaunay triangulation
tri = Delaunay(mean_coord_array)
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
        dist = haversine_dist_from_coords(coord_1[1], coord_1[0], coord_2[1], coord_2[0])
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

print("Transmission edges: ", edges)

print("Total length of transmission line used: ", total_transmission_dist)

# --------------- SAVE ---------------- #

T_save = nx.relabel_nodes(T, loc)
T_dir = T_save.to_directed()
nx.write_shp(T_dir, tree_shapefiles + '/transmission')

# --------------- PLOTS --------------- #

# Get the x range (i.e. indices of points) for plotting
x_range = list(range(1, no_points+1))

# Interpolate the distance data using Univariate Spline for plotting
#spline = UnivariateSpline(x_range, distance_km, s=0, k=4)

# Plot the ordered distances, spline interpolation, and knee point
print("Plotting nearest neighbour distances, univariate spline, and knee point...")
f, ax = plt.subplots(1, figsize=(9, 9))
plt.plot(x_range, distance_km, marker='o', label='Distance to Kth nearest point')
#plt.plot(x_range, spline(x_range), '-', label='Univariate spline')
# plt.plot(distToLine, label="Distance from point to vector spanning curve")
plt.plot([knee_index], knee_value, marker='o', markersize=16, color="red", label='Knee point')
ax.legend(prop={'size': 16})
# plt.title('Distance from each point to Kth nearest neighbour, sorted ascending')
plt.xlabel('Points', fontsize=16)
plt.ylabel('Distance to Kth nearest neighbour (km)', fontsize=16)
plt.xticks(size=16)
plt.yticks(size=16)
plt.show(block=False)
plt.savefig(knee_graph)

# Plot the clusters found
print("Plotting DBSCAN clustering result...")

# Create a new figure
fig, ax = plt.subplots(figsize=(5, 5))

# Loop through labels and plot each cluster
for label in cluster_ids:
    # Add data points - make outliers (-1) black
    if label == -1:
        plt.scatter(x=OSM_data.loc[OSM_data['cluster_index_1'] == label, 'lon'],
                    y=OSM_data.loc[OSM_data['cluster_index_1'] == label, 'lat'],
                    s=20, color='black', alpha=1) #, edgecolors='black'
    else:
        plt.scatter(x=OSM_data.loc[OSM_data['cluster_index_1'] == label, 'lon'],
                    y=OSM_data.loc[OSM_data['cluster_index_1'] == label, 'lat'],
                    s=20, alpha=1) # , edgecolors='black'

# plt.title('Clusters (DBSCAN: eps = knee point, minpts = 5)')
plt.xlabel('Longitude ($^\circ$)', fontsize=16)
plt.ylabel('Latitude ($^\circ$)', fontsize=16)
ax.set_aspect('equal', 'box')
plt.xticks(size=16)
plt.yticks(size=16)
fig.set_size_inches(16, 16)
plt.show(block=False)
plt.savefig(cluster_graph)

# Plot Minimum Spanning Trees made from Delaunay Triangulation
plt.figure()
for key in min_span_tree.keys():
    if key == -1:
        plt.scatter(x=OSM_data.loc[OSM_data['cluster_index_1'] == key, 'lon'],
                    y=OSM_data.loc[OSM_data['cluster_index_1'] == key, 'lat'],
                    s=20, color='black', alpha=1)
        continue
    nx.draw_networkx(min_span_tree[key], pos=locations[key], with_labels=False, node_size=15)
nx.draw_networkx(T, pos=loc, with_labels=False, node_size=15, node_color='r')
plt.title('Minimum Spanning Trees of Delaunay Graph \n (Edge Weight = Haversine Distance)')
plt.xlabel('Longitude ($^\circ$)')
plt.ylabel('Latitude ($^\circ$)')
axes = plt.gca()

plt.show()

print("Plots closed!")
