# Clustering process to group homes for rural electrification
# PSCC 2020 Submission
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk
# Date: 2019-09-24

import sys
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import geopandas as gp
import sklearn.cluster as cluster
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import UnivariateSpline
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt

# --------------- PARAMETERS --------------- #

DBSCAN_MINPTS = 5
K = 5

# --------------- CONSTANTS --------------- #

R_EARTH_KM = 6371

# --------------- FILEPATHS TO DEFINE --------------- #

# Filepath to OpenStreetMap building footprints data as GeoJSON or ShapeFile, trimmed to region of interest
# This can be downloaded from http://download.geofabrik.de/africa.html as a ShapeFile
buildings = 'FILL ME WITH FILEPATH TO OSM BUILDING FOOTPRINTS'

# Filepaths to save final results
clustering_results = 'FILL ME WITH PATH WHERE YOU WANT CLUSTERING RESULTS SAVED.geojson'
console_output = 'FILL ME WITH PATH WHERE YOU WANT CONSOLE OUTPUT SAVED.txt'
knee_graph = 'FILL ME WITH PATH WHERE YOU WANT KNEE GRAPH SAVED.png'
cluster_graph = 'FILL ME WITH PATH WHERE YOU WANT CLUSTER GRAPH SAVED.png'

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


# Haversine formula for kilometer distance between two lat/long points
# Implementation based on the following tutorial:
# https://www.geeksforgeeks.org/program-distance-two-points-earth/
def distance_from_coordinates(lat1, lat2, lon1, lon2):
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
sys.stdout = open(console_output, 'w')

# Import data as GeoDataFrame
print('Reading building footprint data into a GeoDataFrame...')
OSM_data = gp.read_file(buildings)
print('GeoDataFrame is ready!')

# Pre-process data to get lat/long array (X) and number of points (no_points)
print('Pre-processing building footprint data...')
# Get the number of rows (points) in the GeoDataFrame
no_points = OSM_data.shape[0]
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
OSM_data = OSM_data.drop(['centroid'], axis=1)

print("Saving clustering results as GeoJSON...")
# Save OSM data with both levels of clustering included as a GeoJSON
OSM_data.to_file(clustering_results, driver='GeoJSON')
print("Results saved! Location: ", clustering_results)

print('Execution time: ', datetime.now() - start_time)

# --------------- PLOTS --------------- #

# Get the x range (i.e. indices of points) for plotting
x_range = list(range(1, no_points+1))

# Get unique cluster label values
cluster_ids = OSM_data['cluster_index_1'].unique()

# Interpolate the distance data using Univariate Spline for plotting
spline = UnivariateSpline(x_range, distance_km, s=0, k=4)

# Plot the ordered distances, spline interpolation, and knee point
print("Plotting nearest neighbour distances, univariate spline, and knee point...")
f, ax = plt.subplots(1, figsize=(9, 9))
plt.plot(x_range, distance_km, marker='o', label='Distance to Kth nearest point')
plt.plot(x_range, spline(x_range), '-', label='Univariate spline')
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

plt.show()

print("Plots closed!")
