# Clustering of home footprint centroids with HAC
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import geopandas as gp
import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Define filename for SLE OSM building data sample and SLE district bounds
FILENAME_CENTROIDS = 'PSCCAbstract_Centroids.geojson'
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

# Break out lat and long into separate columns
centroids['lon'] = centroids.geometry.x
centroids['lat'] = centroids.geometry.y

# HIERARCHICAL AGGLOMERATIVE CLUSTERING

# Get only lat and long columns from centroids geodataframe and make this into a numpy array
X = centroids.drop(['name', 'type', 'code', 'fclass', 'osm_id', 'geometry'], axis=1).to_numpy()

# Do HAC, fit to our data
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.001, affinity='euclidean', linkage='average').fit(X)

# Print cluster labels
print(clustering.labels_)

# Plot the result
f, ax = plt.subplots(1, figsize=(9, 9))
centroids.assign(cl=clustering.labels_).plot(column='cl', categorical=True, legend=True, linewidth=0.1, edgecolor='white', ax=ax)
ax.set_axis_off()
plt.show()

# Save centroids with cluster number as .shp
centroids['cluster_index'] = clustering.labels_
# print(centroids.head())
centroids.to_file("HAC_001_average.geojson", driver='GeoJSON')

plt.scatter(X[:, 0], X[:, 1], c=hac.labels_, cmap='rainbow')
plt.show()
