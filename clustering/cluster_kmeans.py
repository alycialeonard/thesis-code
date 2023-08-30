# Clustering of home footprint centroids with K-means
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import geopandas as gp
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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
print(centroids.head())
bounds = gp.read_file(BOUNDS_PATH)

# Plot centroids on basemap of Sierra Leone
fig, ax = plt.subplots()
base = bounds.plot(ax=ax, color='darkblue')
centroids.plot(ax=base, color='red', marker='.')
ax.set_title("Test Dataset: Sierra Leone OSM Building Centroids Sample", fontsize=12)
plt.show()

# Break out lat and long into separate columns from geometry column
centroids['lon'] = centroids.geometry.x
centroids['lat'] = centroids.geometry.y

# K-MEANS CLUSTERING

# Get only lat and long columns from centroids dataframe and make this into a numpy array
X = centroids.drop(['name', 'type', 'code', 'fclass', 'osm_id', 'geometry'], axis=1).to_numpy()

# Do k-means clustering (adjust n_clusters) fitted to lat/lon
clustering = KMeans(n_clusters=5).fit(X)

# Print cluster labels
print(clustering.labels_)

# Save centroids with cluster number as .shp
centroids['cluster_index'] = clustering.labels_
print(centroids.head())
centroids.to_file("centroids_kmeans_5.shp")

# Plot the result
f, ax = plt.subplots(1, figsize=(9, 9))
centroids.assign(cl=clustering.labels_).plot(column='cl', categorical=True, legend=True, linewidth=0.1, edgecolor='white', ax=ax)
ax.set_axis_off()
plt.show()
