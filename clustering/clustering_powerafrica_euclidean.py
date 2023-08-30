# Clustering + network design for rural electrification (edited from PowerAfrica)
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk
# Date: 2021-04-09
# Cluster results analysis added 2021-06-15

from datetime import datetime
import pandas as pd
import geopandas as gp
import networkx as nx
import sklearn.cluster as cluster
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial import distance
from kneed import KneeLocator

# --------------- PARAMETERS --------------- #

# Minimum number of homes in a DBSCAN cluster
DBSCAN_MINPTS = 5

# Number of nearest neighbours to consider in knee-point calculations
K = 5

# Sensitivity in knee-finding algorithm (see: https://kneed.readthedocs.io/en/stable/api.html)
S = 4

# Whether legal under-grid distance will be considered, and distance to consider if so
under_grid_considered = False
under_grid = 200

# --------------- FILEPATHS TO DEFINE --------------- #

# Filepath to OpenStreetMap building footprints data as GeoJSON or ShapeFile, trimmed to region of interest
# This can be downloaded from http://download.geofabrik.de/africa.html as a ShapeFile
buildings = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\' \
            'gis_osm_buildings_a_free_1_kenema_district.geojson'

# Filepaths to save results
clustering_results = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\' \
                     'result_3857_2\\clustering_results.geojson'
clustering_results_no_outliers = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\' \
                                 'result_3857_2\\clustering_results_no_outliers.geojson'
analysis_results_csv = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\' \
                       'result_3857_2\\analysis_results.csv'
analysis_results = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\' \
                   'result_3857_2\\analysis_results.geojson'
cluster_gdf_path = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\result_20210616\\' \
                   'minpts_5_eps_knee\\results_analysis\\households_per_cluster.csv'
cluster_gdf_value_counts_path = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\' \
                                'result_20210616\\minpts_5_eps_knee\\results_analysis\\clusters_per_cluster_size.csv'

console_output = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\' \
                 'result_3857_2\\output.txt'
knee_graph = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\' \
             'result_3857_2\\knee_graph.png'
cluster_graph = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\' \
                'result_3857_2\\cluster_graph.png'
tree_shapefiles = 'C:\\Users\\alyci\\Documents\\DPhil\\Working\\sample_areas\\kenema_district\\' \
                  'result_3857_2\\shapefile_exports'

# --------------- PRE-PROCESSING --------------- #

# Get process start time
start_time = datetime.now()

# Uncomment the following to set stdout to print to output file
# sys.stdout = open(console_output, 'w')

# Import data as GeoDataFrame
print('Reading building footprint data into a GeoDataFrame...')
OSM_data = gp.read_file(buildings)
print('GeoDataFrame is ready!')

# Pre-process data to get array of x,y coordinates and number of points (no_points)
print('Pre-processing building footprint data...')
# Get the number of rows (points) in the GeoDataFrame
no_points = OSM_data.shape[0]
print("Number of points:", no_points)
# Convert to EPSG 3857 (projected) to prevent errors in centroid calculations (OSM data is originally 4326)
OSM_data = OSM_data.to_crs("EPSG:3857")
# Get centroid points of building polygons
OSM_data['centroid'] = OSM_data['geometry'].centroid
# Break out x and y values into separate columns of GeoDataFrame
OSM_data['x'] = OSM_data.centroid.x
OSM_data['y'] = OSM_data.centroid.y
# Get x and y columns from the GeoDataFrame and convert into a numpy array
coordinates = OSM_data.drop(['name', 'type', 'code', 'fclass', 'osm_id', 'geometry', 'centroid'], axis=1).to_numpy()
print('Array of building centroids is ready for analysis!')

# --------------- NEAREST NEIGHBOURS & CLUSTERING --------------- #

# Get n nearest neighbours to each point (fit to points and query on points, nearest neighbour of each is itself)
print('Nearest neighbors calculation on centroids starting...')
nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(coordinates)
# Get distances (in meters) and indices
distances_m, indices = nbrs.kneighbors(coordinates)
print('Nearest neighbors complete!')

# Sort distances to nth neighbour for all points in ascending order
distance_m = np.asarray(sorted(distances_m[:, K-1]))
# Get an array of x values (0 to no_points-1)
x_range = np.arange(no_points)
# Get the knee point of the ordered distances
print('Calculating knee-point of nearest neighbour distances...')
kneedle = KneeLocator(x_range, distance_m, S=S, curve="convex", direction="increasing")
knee_index = kneedle.elbow
knee_value = kneedle.knee_y
print("Knee of the curve: index = ", knee_index, ", value = ", knee_value, " meters.")
kneedle.plot_knee()
plt.show()


# If under-grid distance is being considered, and the knee value is less than it, reassign it to under-grid
if (knee_value < under_grid) and under_grid_considered:
    knee_value = under_grid
    print('Knee value is less than legal under-grid distance - bumping up to under-grid distance')

# Cluster data using DBSCAN with eps=knee_value in deg
print("Clustering building footprint centroids using DBSCAN (eps = knee point)...")
clustering = cluster.DBSCAN(eps=knee_value, min_samples=DBSCAN_MINPTS, algorithm='ball_tree').fit(coordinates)
print("Clustering complete!")

# Append cluster labels to the gdf of data
OSM_data['cluster_index_1'] = clustering.labels_
# Get number of DBSCAN clusters created (0 to N)
N = np.amax(clustering.labels_)

# Only one geometry column allowed in GeoJSON so drop centroid to save - still preserved in x and y columns
OSM_data_save = OSM_data.drop(['centroid'], axis=1)
# Uncomment the following line to save a separate file for non-outlier points
OSM_data_save_no_outliers = OSM_data_save[OSM_data_save.cluster_index_1 != -1]

print("Saving clustering results as GeoJSON...")
# Save OSM data with both levels of clustering included as a GeoJSON
OSM_data_save.to_file(clustering_results, driver='GeoJSON')
# Uncomment the following line to save a separate file for non-outlier points
OSM_data_save_no_outliers.to_file(clustering_results_no_outliers, driver='GeoJSON')
print("Results saved! Location: ", clustering_results)

clust_exec_time = datetime.now() - start_time
print('Execution time for pre-processing and clustering: ', clust_exec_time)

# --------------- CLUSTER RESULTS ANALYSIS --------------- #

# Output series of # of homes in each cluster id (cluster id is index)
cluster_gdf = OSM_data_save_no_outliers.groupby(['cluster_index_1'], sort=False).size()
# Save this and number of clusters per cluster size as csv
cluster_gdf.to_csv(cluster_gdf_path)
cluster_gdf.value_counts().to_csv(cluster_gdf_value_counts_path)

# Load number of clusters per cluster size as df
cluster_sizes_df = pd.read_csv(cluster_gdf_value_counts_path)
cluster_sizes_df = cluster_sizes_df.sort_values('Cluster size (households)')
cluster_sizes_df.plot(x="Cluster size (households)", y="Number of clusters")
plt.show()

# --------------- DISTRIBUTION GRIDS --------------- #

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

    # Skip cluster_id = -1 (cluster of outliers, don't need stats or network)
    if cluster == -1:
        continue

    # Get a subset of gdf for the cluster being considered
    subset = gp.GeoDataFrame(OSM_data.loc[OSM_data['cluster_index_1'] == cluster])

    # Get mean point of each cluster
    mean_coord[cluster] = [subset['x'].mean(), subset['y'].mean()]
    # Get and record the number of points in that cluster
    no_points_cluster = subset.shape[0]
    building_count[cluster] = no_points_cluster

    # Get x and y columns from the GeoDataFrame and convert into a numpy array
    subset_coords = subset.drop(['name', 'type', 'code', 'fclass', 'osm_id', 'geometry', 'centroid', 'cluster_index_1'],
                                axis=1).to_numpy()
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
        dist = distance.euclidean(coord_1, coord_2)
        d = d + dist
    total_conductor_length_radial[cluster] = d

print("Lengths of conductor used in each community MST cluster: ", total_conductor_length)
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
nx.write_shp(T_dir, tree_shapefiles + '/transmission')

# --------------- PLOTS --------------- #

# Get the x range (i.e. indices of points) for plotting
# x_range = list(range(1, no_points+1))

# Plot the ordered distances and knee point
print("Plotting nearest neighbour distances and knee point...")
f, ax = plt.subplots(1, figsize=(9, 9))
plt.plot(x_range.tolist(), distance_m, marker='o', label='Distance to Kth nearest point')
# plt.plot(distToLine, label="Distance from point to vector spanning curve")
plt.plot([knee_index], knee_value, marker='o', markersize=16, color="red", label='Knee point')
ax.legend(prop={'size': 16})
# plt.title('Distance from each point to Kth nearest neighbour, sorted ascending')
plt.xlabel('Points', fontsize=16)
plt.ylabel('Distance to Kth nearest neighbour (m)', fontsize=16)
plt.xticks(size=16)
plt.yticks(size=16)
plt.show(block=False)
plt.savefig(knee_graph)

# Plot the clusters found
print("Plotting DBSCAN clustering result...")
fig, ax = plt.subplots(figsize=(5, 5))
# Loop through labels and plot each cluster
for label in cluster_ids:
    # Add data points - make outliers (-1) black
    if label == -1:
        plt.scatter(x=OSM_data.loc[OSM_data['cluster_index_1'] == label, 'x'],
                    y=OSM_data.loc[OSM_data['cluster_index_1'] == label, 'y'],
                    s=20, color='black', alpha=1)  # edgecolors='black'
    else:
        plt.scatter(x=OSM_data.loc[OSM_data['cluster_index_1'] == label, 'x'],
                    y=OSM_data.loc[OSM_data['cluster_index_1'] == label, 'y'],
                    s=20, alpha=1)  # edgecolors='black'
# plt.title('Clusters (DBSCAN: eps = knee point, minpts = 5)')
plt.xlabel('Distance (m)', fontsize=16)
plt.ylabel('Distance (m)', fontsize=16)
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
        plt.scatter(x=OSM_data.loc[OSM_data['cluster_index_1'] == key, 'x'],
                    y=OSM_data.loc[OSM_data['cluster_index_1'] == key, 'y'],
                    s=20, color='black', alpha=1)
        continue
    nx.draw_networkx(min_span_tree[key], pos=locations[key], with_labels=False, node_size=15)
nx.draw_networkx(T, pos=loc, with_labels=False, node_size=15, node_color='r')
# plt.title('Minimum Spanning Trees of Delaunay Graph \n (Edge Weight = Euclidean Distance)')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
axes = plt.gca()

plt.show()

print("Plots closed!")
