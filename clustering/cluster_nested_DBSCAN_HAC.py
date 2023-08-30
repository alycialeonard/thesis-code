# Clustering of home footprint centroids with a nested process of DBSCAN then HAC.
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import geopandas as gp
import pandas as pd
import os
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import numpy as np
import cluster_tuning as tune

# DEFINE PARAMETERS:

# Filename of building polygon (or centroid) OpenStreetMap data, GeoJSON format:
FILENAME = "PSCCAbstract_Centroids.geojson"

# DBSCAN top-level clustering hyperparameters (to be replaced with heuristic calculations)
DBSCAN_EPS = 0.001
DBSCAN_MINPTS = 5

# HAC sub-cluster clustering hyperparameters (to be replaced with heuristic calculations)
# HAC_NCLUSTERS = 3
HAC_LINKAGE = 'average'
HAC_DISTANCE_THRESHOLD = 0.0006

# IMPORT DATA:

# Get project path (one level up from script, script being run from script folder)
#os.chdir('..')
PROJECT_PATH = os.getcwd()

# Get data path
DATA_PATH = os.path.join(PROJECT_PATH, 'data', 'clustering_toy_dataset', FILENAME)

# Import data as GeoDataFrame
gdf = gp.read_file(DATA_PATH)

# PROCESS DATA:

# Get centroid points of building polygons
gdf['centroid'] = gdf['geometry'].centroid

# Break out lat and long into separate columns
gdf['lon'] = gdf.centroid.x
gdf['lat'] = gdf.centroid.y

# Get only lat and long columns from gdf and convert into a numpy array for clustering
X = gdf.drop(['name', 'type', 'code', 'fclass', 'osm_id', 'geometry', 'centroid'], axis=1).to_numpy()

# CLUSTER DATA:

# Run heuristic for optimising clustering parameters
# DBSCAN_EPS_2, DBSCAN_MINPTS_2 = tune.hyperparameter_tuning_dbscan(DATA_PATH)

# Cluster lat/long data using DBSCAN (microgrid/national grid/SHS segregation)
clustering = cluster.DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MINPTS).fit(X)

# Append cluster labels to gdf
gdf['cluster_index_1'] = clustering.labels_

# Get number of clusters created (0 to N to N+1 clusters exist)
N = np.amax(clustering.labels_)

# Make empty dataframe for results
results = pd.DataFrame()

# Internally cluster each cluster subset with HAC
for i in range(N+1):
    # Get a subset of gdf for the cluster being considered
    subset = gp.GeoDataFrame(gdf.loc[gdf['cluster_index_1'] == i])
    # Save subset as shapefile for tuning heuristic
    subset_save = subset.drop('centroid', axis=1)
    subset_save.to_file('temp.geojson', driver='GeoJSON')
    # Get just lat and long as numpy array for this cluster
    X2 = subset.drop(['name', 'type', 'code', 'fclass', 'osm_id', 'geometry', 'centroid', 'cluster_index_1'], axis=1).to_numpy()
    # Run heuristic for optimising cluster parameters
    # HAC_NCLUSTERS_2, HAC_LINKAGE_2 = tune.hyperparameter_tuning_hac('temp.GeoJSON')
    # Cluster these points using HAC
    # clustering = cluster.AgglomerativeClustering(n_clusters=HAC_NCLUSTERS, linkage=HAC_LINKAGE).fit(X2)
    clustering = cluster.AgglomerativeClustering(n_clusters=None, compute_full_tree=True, linkage=HAC_LINKAGE, distance_threshold=HAC_DISTANCE_THRESHOLD).fit(X2)
    # Put output cluster labels into a dataframe r
    d = {'cluster_index_2': clustering.labels_}
    r = pd.DataFrame(data=d)
    # Append lon and lat from X2 centroids column to r - r now has cluster index, lat and long
    r['centroid'] = X2.tolist()
    r['lon'] = r['centroid'].str[0]
    r['lat'] = r['centroid'].str[1]
    r = r.drop('centroid', axis=1)
    # Append cluster_index_2 to results df
    results = results.append(r, ignore_index=True)

# Merge results into original GDF on lat and lon. Use outer merge so that outliers with no cluster_label_2 are kept
gdf2 = pd.merge(gdf, results, on=['lat', 'lon'], how='outer')

# Fill "NaN" results in cluster_index_2 for outliers with -1
gdf2['cluster_index_2'].fillna(value=-1, inplace=True)

# SAVE RESULT:

# Get rid of centroids, keep geometry as polygons for shapefile (only one geometry permitted in shapefile)
gdf_save = gdf2.drop('centroid', axis=1)
# Save as GeoJSON
gdf_save.to_file('cluster_process_test.geojson', driver='GeoJSON')

# PLOT RESULT:

# In order to plot centroids instead of buildings, drop "geometries", keep centroids, rename centroids as geometries
gdf2 = gdf2.drop('geometry', axis=1)
gdf2 = gdf2.rename(columns={'centroid': 'geometry'})

# Subplot 1: top-level clusters
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 9))
gdf2.plot('cluster_index_1', categorical=True, linewidth=0.1, ax=ax1)
ax1.set_title('Top-level DBSCAN Clustering')
ax1.set(xlabel='Longitude', ylabel='Latitude')

# Subplot 2: second-level clusters
gdf2.plot('cluster_index_2', categorical=True, linewidth=0.1, ax=ax2)
ax2.set_title('Second-level HAC Clustering')
ax2.set(xlabel='Longitude', ylabel='Latitude')

plt.show()





