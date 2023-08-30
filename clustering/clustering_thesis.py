# Clustering process for last chapter of thesis and Katiri case study

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

# --------------- PARAMETERS --------------- #

# Minimum number of homes in a DBSCAN cluster
DBSCAN_MINPTS = 5

# Number of nearest neighbours to consider in knee-point calculations
K = 5

# Sensitivity in knee-finding algorithm (see: https://kneed.readthedocs.io/en/stable/api.html)
# 1 is suggested default in original paper
S = 1

# Whether legal under-grid distance will be considered, and distance to consider if so
under_grid_considered = False
under_grid = 200  # in meters

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

# Import data as GeoDataFrame
print('Reading location data into a GeoDataFrame...')
gdf = gp.read_file(data)
print('GeoDataFrame is ready!')

# Pre-process data to get array of x,y coordinates and number of points (no_points)
print('Pre-processing building footprint data...')
# Get the number of rows (points) in the GeoDataFrame
no_points = gdf.shape[0]
print("Number of points:", no_points)
# Convert to EPSG 3857 (projected) to prevent errors in centroid calculations (data is originally 4326)
gdf = gdf.to_crs("EPSG:3857")

# Drop all the useless columns
gdf = gdf.drop(['subject_id', 'x', 'y', 'width', 'height', 'angle', 'metadata', 'filename'], axis=1)
# Break out x and y values into separate columns of GeoDataFrame
gdf['x'] = gdf.geometry.x
gdf['y'] = gdf.geometry.y
# Convert x and y into numpy array
coordinates = gdf.drop(['geometry'], axis=1).to_numpy()
print('Array of building centroids is ready for analysis!')

# --------------- NEAREST NEIGHBOURS & CLUSTERING --------------- #

# Get n nearest neighbours to each point (fit to points and query on points, nearest neighbour of each is itself)
print('Nearest neighbors calculation on centroids starting...')
nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(coordinates)
# Get distances (in meters) and indices
distances, indices = nbrs.kneighbors(coordinates)
print('Nearest neighbors complete!')

# If we are using the under grid distance and ignoring natural variation
if under_grid_considered:
    knee_value = under_grid
    print('Forcing eps = defined under_grid value (' + str(under_grid) + ') regardless of knee.')
else:  # If not, get the knee and set the epsilon value to that distance
    # Sort distances to nth neighbour for all points in ascending order
    dist = np.asarray(sorted(distances[:, K-1]))
    # Get an array of x values (0 to no_points-1)
    x_range = np.arange(no_points)
    # Get the knee point of the ordered distances
    print('Calculating knee-point of nearest neighbour distances...')
    kneedle = KneeLocator(x_range, dist, S=S, curve="convex", direction="increasing")
    knee_index = kneedle.elbow
    knee_value = kneedle.knee_y
    print("Knee of the curve: index = ", knee_index, ", value = ", knee_value, " meters.")
    kneedle.plot_knee()
    plt.legend('',frameon=False)
    plt.xlabel("Homes")
    plt.ylabel("Distance to K = 5th nearest neighbour (m)")
    plt.xlim(290,340)
    plt.ylim(0,200)
    plt.grid(True, which='major')
    plt.minorticks_on()
    plt.title("")
    plt.show()
    #plt.savefig(knee_graph)

# Cluster data using DBSCAN with eps=knee_value in deg
print("Clustering building footprint centroids using DBSCAN...")
clustering = cluster.DBSCAN(eps=knee_value, min_samples=DBSCAN_MINPTS, algorithm='ball_tree').fit(coordinates)
print("Clustering complete!")

# Append cluster labels to the gdf of data
gdf['cluster_index_1'] = clustering.labels_
# Get number of DBSCAN clusters created (0 to N)
N = np.amax(clustering.labels_)

# --------------- SAVE DATA --------------- #

# Save data, with clustering index included, as a GeoJSON
print("Saving clustering results as GeoJSON...")
# gdf.to_file(clustering_results, driver='GeoJSON')

# Uncomment the following to get and save save a separate file for non-outlier points
gdf_no_outliers = gdf[gdf.cluster_index_1 != -1]
# gdf_no_outliers.to_file(clustering_results_no_outliers, driver='GeoJSON')

print("Results saved! Location: ", clustering_results)

clust_exec_time = datetime.now() - start_time
print('Execution time for pre-processing and clustering: ', clust_exec_time)

# Get unique cluster label values
cluster_ids = gdf['cluster_index_1'].unique()

# --------------- PLOTS --------------- #

# Plot the clusters found
print("Plotting DBSCAN clustering result...")
fig, ax = plt.subplots(figsize=(5, 5))
# Loop through labels and plot each cluster
for label in cluster_ids:
    # Add data points - make outliers (-1) black
    if label == -1:
        plt.scatter(x=gdf.loc[gdf['cluster_index_1'] == label, 'x'],
                    y=gdf.loc[gdf['cluster_index_1'] == label, 'y'],
                    s=20, color='black', alpha=1)  # edgecolors='black'
    else:
        plt.scatter(x=gdf.loc[gdf['cluster_index_1'] == label, 'x'],
                    y=gdf.loc[gdf['cluster_index_1'] == label, 'y'],
                    s=20, alpha=1)  # edgecolors='black'
# plt.title('Clusters (DBSCAN: eps = knee point, minpts = 5)')
plt.xlabel('Distance (m)', fontsize=16)
plt.ylabel('Distance (m)', fontsize=16)
ax.set_aspect('equal', 'box')
plt.xticks(size=16)
plt.yticks(size=16)
fig.set_size_inches(16, 16)
plt.show(block=False)
#plt.savefig(cluster_graph)
