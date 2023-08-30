# Clustering of home footprint centroids with HAC (own script)
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import geopandas as gp
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
import math
from itertools import combinations


# Helper function for the distance between two points (passed as tuples)
def dist(p1, p2):
    [x1, y1], [x2, y2] = p1, p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# Define filename for SLE OSM building data sample and SLE district bounds
FILENAME_CENTROIDS = 'gis_osm_buildings_a_free_1_EvenSmallerSample_Centroids.geojson'
FILENAME_BOUNDS = 'sle_admbnda_adm2_1m_gov_ocha_20161017.shp'

# Move up one directory layer from clustering to GeoDESA and get path
os.chdir('..')
CWD_PATH = os.getcwd()

# Define path to centroid and bounds
CENTROIDS_PATH = os.path.join(CWD_PATH, 'data', 'clustering_toy_dataset', FILENAME_CENTROIDS)
BOUNDS_PATH = os.path.join(CWD_PATH, 'data', 'SLE_bounds', 'sle_admbnda_adm2_1m_gov_ocha', FILENAME_BOUNDS)

# Read data files. They will become GeoDataFrames.
centroids = gp.read_file(CENTROIDS_PATH)
bounds = gp.read_file(BOUNDS_PATH)

# Plot centroids on basemap of Sierra Leone
fig, ax = plt.subplots()
base = bounds.plot(ax=ax, color='darkblue')
centroids.plot(ax=base, color='red', marker='.')
ax.set_title("Test Dataset: Sierra Leone OSM Building Centroids Sample", fontsize=12)
plt.show()

# Get rid of centroid name, type, code, and fclass columns; we will work with osm_id and geometry columns
centroids = centroids.drop(['name', 'type', 'code', 'fclass'], axis=1)

# Break out lat and long into separate columns
centroids['lon'] = centroids.geometry.x
centroids['lat'] = centroids.geometry.y

# Make list of tuples of lat/long points. Convert to list of lists, and then to numpy array.
X = list(zip(centroids['lat'], centroids['lon']))
X = [list(point) for point in X]
X = np.array(X)

# Define criteria for a successful cluster

# 0.01 degrees is approx 1km at the equator
max_avg_distance = 0.0025

# Furthest a point in a cluster can be from its nearest neighbour before it is excluded from the cluster
max_nearest_neighbour_distance = 0.0003

# CLUSTERING ALGORITHM

# Initialize the number of clusters to make as 1
no_clusters = 1
# Instantiate a list with one element "False" to start record clustering results
results = [False]
# While the results aren't all successful (i.e. at least one "False" result meaning one cluster too spread)
while not(all(results)):
    if no_clusters == 1:
        # If you're just starting, print that you're starting
        print("\nStarting clustering algorithm:\n")
    else:
        # If not, print last cycle's unsuccessful results
        print("Results from last cycle:", results)
        print("Unsuccessful: Increasing number of clusters to create.\n")
    # Clear the last results and then re-cluster the points with new no_clusters
    results = []
    cluster = AgglomerativeClustering(n_clusters=no_clusters, affinity='euclidean', linkage='average')
    cluster.fit_predict(X)
    print("Clustering complete!")
    print("Number of clusters made (no_clusters):", no_clusters)
    print("Clusters assigned:", cluster.labels_)
    # For each cluster that was made, test the average distance between points
    i = 0
    while i < no_clusters:
        print("Cluster being considered (i):", i)
        # Get indices of points in this cluster within full points list
        c_indices = np.where(cluster.labels_ == i)
        # Access all the points in this cluster by their indices, put in list of arrays "c"
        c = [X[index] for index in c_indices]
        # Reformat as list of lists in inner comprehension, and then to list of tuples in outer comprehension
        c = [tuple(elements) for elements in [elements.tolist() for elements in c][0]]
        print("Points in this cluster:", c)
        # CHECK CLUSTER CRITERIA:
        # For each point in cluster, check and record distance to nearest neighbour
        results_distance = []
        for point in c:
            # Get distance from point to each point in the cluster
            distance_matrix = distance.cdist(np.asarray([point]), np.asarray(c))
            # Get the minimum (non-zero) distance
            distance_to_closest = np.min(distance_matrix[np.nonzero(distance_matrix)])
            # print(distance_to_closest)
            # If it's too far from it's nearest neighbour, remove from cluster!
            if distance_to_closest > max_nearest_neighbour_distance:
                results_distance.append(False)
            else:
                results_distance.append(True)
        # Grab points which passed nearest neighbour test:
        c_updated_indices = np.where(results_distance == True) # Fix this
        c_updated = [X[index] for index in c_updated_indices]
        print("Points in this cluster, exluding outliers:", c_updated)
        # Get distances between all combinations of two points in the cluster
        distances = [dist(p1, p2) for p1, p2 in combinations(c, 2)]
        # If there is at least one distance (i.e. the cluster has more than one point), get avg distance
        # If there is only one point, assign 0 (prevent division by 0)
        if len(distances) > 0:
            avg_distance = sum(distances) / len(distances)
        else:
            avg_distance = 0
        print("Average distance between points in this cluster:", avg_distance)
        # Add the result as False if the avg distance is above a threshold, and true if not
        if avg_distance > max_avg_distance:
            results.append(False)
        else:
            results.append(True)
        # Got to next cluster
        i = i+1
    # Increment the number of clusters to try
    no_clusters = no_clusters + 1

# Plot the successful clusters
print("Success! Plotting final clusters.")
plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap='rainbow')
plt.show()

# Save centroids with cluster number as .shp
centroids['cluster_index'] = cluster.labels_
print(centroids.head())
centroids.to_file("centroids.shp")
