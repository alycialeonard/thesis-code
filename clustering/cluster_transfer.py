# Clustering and analysis process for Transfer of Status + PSCC 2020 paper
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk
# Date: 2019-08-19

import sys
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from osgeo import gdal
import pandas as pd
import geopandas as gp
from rasterstats import zonal_stats
from shapely.geometry import shape, Point
import json
from skimage.filters import threshold_otsu
import sklearn.cluster as cluster
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import UnivariateSpline
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# --------------- PARAMETERS --------------- #

DBSCAN_MINPTS = 5
K = 5


# --------------- CONSTANTS --------------- #

R_EARTH_KM = 6371


# --------------- FILEPATHS --------------- #

# Filepath to OpenStreetMap building footprints data trimmed to region of interest
# Downloaded from http://download.geofabrik.de/africa.html
# buildings = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\openstreetmap\\makueni_case_osm_buildings.geojson'
buildings = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\openstreetmap\\kenema_case_osm_buildings.geojson'
# buildings = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\openstreetmap\\bo_case_osm_buildings.geojson'
# buildings = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\openstreetmap\\bidibidi_case_osm_buildings.geojson'

# Filepaths to rasters
# Luminance raster applies luminance formula from ITU BT.709 https://www.itu.int/rec/R-REC-BT.709
# to NASA Black Marble night lights data at 15 arc-second (i.e. 0.00416667 deg) resolution
# downloaded from https://earthobservatory.nasa.gov/features/NightLights
# clipped to country to be analyzed using a bounding box enclosing the listed country file
# Wind raster is wind power density in W/m^2 at 9 arc-second (i.e. 0.0025 deg) resolution
# downloaded from https://globalwindatlas.info/
# Solar raster is the photovoltaic power potential in kWh/kWp at 30 arc-second (i.e. 0.00833333 deg) resolution
# downloaded from https://globalsolaratlas.info/

# # Kenya
# luminance_raster = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\nasa_blackmarble\\colour\\BlackMarble_2016_luminance_KEN.tif'
# wind_raster = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\worldbank_wind\\kenya_wind.tif'
# solar_raster = 'C:\\Users\\alyci\\Documents\DPhil\\Transfer of Status\\Transfer_data\worldbank_solar\\Kenya_GISdata_LTAy_YearlySum_GlobalSolarAtlas_GEOTIFF\\PVOUT.tif'

# Sierra Leone
luminance_raster = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\nasa_blackmarble\\colour\\BlackMarble_2016_luminance_SLE.tif'
wind_raster = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\worldbank_wind\\sierraleone_wind.tif'
solar_raster = 'C:\\Users\\alyci\\Documents\DPhil\\Transfer of Status\\Transfer_data\worldbank_solar\\Sierra-Leone_GISdata_LTAy_YearlySum_GlobalSolarAtlas_GEOTIFF\\PVOUT.tif'

# # Uganda
# luminance_raster = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\nasa_blackmarble\\colour\\BlackMarble_2016_luminance_UGA.tif'
# wind_raster = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\worldbank_wind\\uganda_wind.tif'
# solar_raster = 'C:\\Users\\alyci\\Documents\DPhil\\Transfer of Status\\Transfer_data\worldbank_solar\\Uganda_GISdata_LTAy_YearlySum_GlobalSolarAtlas_GEOTIFF\\PVOUT.tif'


# Filepaths to country and regional boundary files (GeoJSON)
# produced by United Nations Office for the Coordination of Humanitarian Affairs
# downloaded from https://data.humdata.org/dataset
# For SLE, regions are districts; for KEN, regions are counties; for UGA, regions are districts.
# For SLE, region_property is 'admin2Name'; For UGA, "ADM1_EN"; For KEN, "ADM1_EN"

# # Kenya
# country = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\country_boundaries\\ken_admbnda_adm0_iebc_20180607\\ken_admbnda_adm0_iebc_20180607_30arcsecondbuffer.geojson'
# regions = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\country_boundaries\\ken_admbnda_adm1_iebc_20180607\\ken_admbnda_adm1_iebc_20180607.geojson'
# region_property = "ADM1_EN"

# Sierra Leone
country = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\country_boundaries\\sle_admbnda_adm0_1m_gov_ocha_20161017\\sle_admbnda_adm0_1m_gov_ocha_20161017_30arcsecondbuffer.geojson'
regions = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\country_boundaries\\sle_admbnda_adm2_1m_gov_ocha\\sle_admbnda_adm2_1m_gov_ocha_20161017.geojson'
region_property = "admin2Name"

# # Uganda
# country = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\country_boundaries\\uga_admbnda_adm0_ubos_v2\\uga_admbnda_adm0_UBOS_v2_30arcsecondbuffer.geojson'
# regions = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\country_boundaries\\uga_admbnda_adm1_ubos_v2\\uga_admbnda_adm1_UBOS_v2.geojson'
# region_property = "ADM1_EN"

# Filepaths to save final results

# Kenema
clustering_results = 'C:\\Users\\alyci\\Documents\\DPhil\\kenema_case_clustering_results.geojson'
analysis_results_csv = 'C:\\Users\\alyci\\Documents\\DPhil\\kenema_case_analysis_results.csv'
analysis_results = 'C:\\Users\\alyci\\Documents\\DPhil\\kenema_case_analysis_results.geojson'
console_output = 'C:\\Users\\alyci\\Documents\\DPhil\\kenema_case_output.txt'
knee_graph = 'C:\\Users\\alyci\\Documents\\DPhil\\kenema_case_knee_graph.png'
cluster_graph = 'C:\\Users\\alyci\\Documents\\DPhil\\kenema_case_cluster_graph.png'

# # Bo
# clustering_results = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\bo_case_clustering_results.geojson'
# analysis_results_csv = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\bo_case_analysis_results.csv'
# analysis_results = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\bo_case_analysis_results.geojson'
# console_output = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\bo_case_output.txt'
# knee_graph = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\bo_case_knee_graph.png'
# cluster_graph = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\bo_case_cluster_graph.png'

# # Bidibidi
# clustering_results = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\bidibidi_case_clustering_results.geojson'
# analysis_results_csv = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\bidibidi_case_analysis_results.csv'
# analysis_results = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\bidibidi_case_analysis_results.geojson'
# console_output = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\bidibidi_case_output.txt'
# knee_graph = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\bidibidi_case_knee_graph.png'
# cluster_graph = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\bidibidi_case_cluster_graph.png'

# # Makueni
# clustering_results = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\makueni_case_clustering_results.geojson'
# analysis_results_csv = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\makueni_case_analysis_results.csv'
# analysis_results = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\makueni_case_analysis_results.geojson'
# console_output = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\makueni_case_output.txt'
# knee_graph = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\makueni_case_knee_graph.png'
# cluster_graph = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\case_study_results\\makueni_case_cluster_graph.png'


# --------------- FUNCTIONS --------------- #

# Find the knee of a set of data following a curve up or down
# Inputs: number of points in dataset, ordered datapoints
# This is based on the Kneedle approach: https://raghavan.usc.edu//papers/kneedle-simplex11.pdf
# Implementation from the IBM Watson tutorial here:
# https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1
def find_knee(n_points, data):
    # Stack all points and get the first point
    allCoord = np.vstack((range(n_points), data)).T
    firstPoint = allCoord[0]
    # Get vector between first and last point i.e. the line
    lineVec = allCoord[-1] - allCoord[0]
    # This is a unit vector in the direction of the line
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))
    # Find the distance between all points and first point
    vecFromFirst = allCoord - firstPoint
    # Split vecFromFirst into parallel/perpendicular comps. Take the norm of perpendicular comp & get distance.
    # Project vecFromFirst onto line by taking scalar product with unit vector in direction of line
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, n_points, 1), axis=1)
    # Multiply scalar product by unit vector
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    # Perpendicular vector to the line is the difference between vecFromFirst and the parallel bit
    vecToLine = vecFromFirst - vecFromFirstParallel
    # Distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    # Knee/elbow is the point with the max distance value
    knee_point_index = np.argmax(distToLine)
    knee_point_value = data[knee_point_index]
    return knee_point_index, knee_point_value


# Get the bounding box of a collection of points
# https://stackoverflow.com/questions/46335488/how-to-efficiently-find-the-bounding-box-of-a-collection-of-points
def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]


# Haversine formula for kilometer distance between two lat/long points
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

# Multiply distances by radius of earth in kilometers (r = 6371) to get in kilometers
# This is so the knee value graph is pretty and in units people know
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
# Centroid data is still preserved in lat and long columns
OSM_data = OSM_data.drop(['centroid'], axis=1)

print("Saving clustering results as GeoJSON...")
# Save OSM data with both levels of clustering included as a GeoJSON
OSM_data.to_file(clustering_results, driver='GeoJSON')
print("Results saved! Location: ", clustering_results)

# Instantiate dictionaries to hold info for each cluster
mean_coord = {}
wind = {}
solar = {}
luminance = {}
bright = {}
boxes = {}
area = {}
building_count = {}
region = {}

print("Reading in complementary data (wind, solar, night-time luminance, regional boundaries)...")
# Read in wind data
wind_gdata = gdal.Open(wind_raster)
wind_gt = wind_gdata.GetGeoTransform()
wind_data = wind_gdata.ReadAsArray().astype(np.float)
# Read in solar PV data
solar_gdata = gdal.Open(solar_raster)
solar_gt = solar_gdata.GetGeoTransform()
solar_data = solar_gdata.ReadAsArray().astype(np.float)
# Read in black marble luminance data
luminance_gdata = gdal.Open(luminance_raster)
luminance_gt = luminance_gdata.GetGeoTransform()
luminance_data = luminance_gdata.ReadAsArray().astype(np.float)
luminance_data[luminance_data == -99] = 0
# Read in the regions vector file
js = json.load(open(regions))
print("Data loaded!")

print("Calculating statistics for raster data...")
# Get stats for solar raster
solar_stats = zonal_stats(country, solar_raster, stats=['min', 'max', 'mean', 'median'])
print('Solar stats: ', solar_stats)
# Get stats for wind raster
wind_stats = zonal_stats(country, wind_raster, stats=['min', 'max', 'mean', 'median'])
print('Wind stats: ', wind_stats)
# Get stats for luminance raster:
luminance_stats = zonal_stats(country, luminance_raster, stats=['min', 'max', 'mean', 'median'])
print('Luminance stats: ', luminance_stats)
# Get Otsu threshold of luminance
otsu_thresh = threshold_otsu(luminance_data)
print('Otsu threshold for luminance raster: ', otsu_thresh)

# Get information about each cluster
# For reference: in the geotransform loaded from GDAL,
# [0] = top left x, [1] = w-e pixel resolution, [2] = 0, [3] = top left y, [4] = 0, [5] = n-s pixel resolution

# Get unique cluster label values
cluster_ids = OSM_data['cluster_index_1'].unique()

print('Gathering data for each cluster from complementary data sources...')
for cluster in cluster_ids:
    # Get mean point of each cluster
    mean_coord[cluster] = OSM_data.loc[OSM_data['cluster_index_1'] == cluster, ['lon', 'lat']].mean()
    # Get the wind value at the mean point of the cluster
    x = int((mean_coord[cluster][0] - wind_gt[0])/wind_gt[1])
    y = int((mean_coord[cluster][1] - wind_gt[3])/wind_gt[5])
    wind[cluster] = wind_data[y, x]
    # Get the solar value at the mean point of the cluster
    x = int((mean_coord[cluster][0] - solar_gt[0])/solar_gt[1])
    y = int((mean_coord[cluster][1] - solar_gt[3])/solar_gt[5])
    solar[cluster] = solar_data[y, x]
    # Get the luminance value at the mean point of the cluster
    x = int((mean_coord[cluster][0] - luminance_gt[0])/luminance_gt[1])
    y = int((mean_coord[cluster][1] - luminance_gt[3])/luminance_gt[5])
    luminance[cluster] = luminance_data[y, x]
    # If luminance > Otsu threshold, mark as likely electrified
    if luminance[cluster] > otsu_thresh:
        bright[cluster] = int(True)
    else:
        bright[cluster] = int(False)
    # Get a subset of gdf for the cluster being considered
    subset = gp.GeoDataFrame(OSM_data.loc[OSM_data['cluster_index_1'] == cluster])
    # Get number of points in that cluster
    no_points_cluster = subset.shape[0]
    building_count[cluster] = no_points_cluster
    # Get just lat and long for points in this cluster
    X2 = subset.drop(['name', 'type', 'code', 'fclass', 'osm_id', 'geometry', 'cluster_index_1'], axis=1).to_numpy()
    # Get bounding box of points in the cluster
    box = bounding_box(X2)
    boxes[cluster] = box
    # Get the area of that bounding box in sqkm
    d1 = distance_from_coordinates(boxes[cluster][0][1], boxes[cluster][1][1], boxes[cluster][0][0], boxes[cluster][0][0])
    d2 = distance_from_coordinates(boxes[cluster][0][1], boxes[cluster][0][1], boxes[cluster][0][0], boxes[cluster][1][0])
    area[cluster] = d1 * d2
    # Check to see which region cluster mean is in
    point = Point(mean_coord[cluster])
    for feature in js['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            region[cluster] = feature['properties'][region_property]

# Close all datasets
wind_gdata = None
solar_gdata = None
luminance_gdata = None

print('Data gathering complete!')

# Make a dataframe from all this data
cluster_summary_df = pd.DataFrame.from_dict(mean_coord, orient='index')
cluster_summary_df["region"] = pd.Series(region)
cluster_summary_df["building_count"] = pd.Series(building_count)
cluster_summary_df["bounding_box_area"] = pd.Series(area)
cluster_summary_df["wind"] = pd.Series(wind)
cluster_summary_df["pv_out"] = pd.Series(solar)
cluster_summary_df["night_luminance"] = pd.Series(luminance)
cluster_summary_df["bright"] = pd.Series(bright)
cluster_summary_df['geometry'] = cluster_summary_df.apply(lambda row: Point(row.lon, row.lat), axis=1)

# Turn it into a GeoDataFrame
cluster_summary_gdf = gp.GeoDataFrame(cluster_summary_df)

# Print the number of outliers before this "cluster" gets dropped
print('Note: ', cluster_summary_gdf.loc[-1, 'building_count'], ' outlier buildings were identified in this analysis.')
print('Removing outlier results...')
cluster_summary_gdf = cluster_summary_gdf.drop(-1, axis=0)

print('Saving analysis results as CSV...')
cluster_summary_gdf.to_csv(path_or_buf=analysis_results_csv)
print('Results saved! Location: ', analysis_results_csv)

print('Saving analysis results as GeoJSON for visualization...')
cluster_summary_gdf.to_file(analysis_results, driver='GeoJSON')
print('Results saved! Location: ', analysis_results)

print('Process execution time: ', datetime.now() - start_time)


# --------------- PLOTS --------------- #

# Get the x range (i.e. indices of points) for plotting
x_range = list(range(1, no_points+1))

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


# Plot clusters with numbers labelled - tutorial https://nikkimarinsek.com/blog/7-ways-to-label-a-cluster-plot-python
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
    # # Add labels
    # plt.annotate(label,a
    #              OSM_data.loc[OSM_data['cluster_index_1'] == label, ['lon', 'lat']].mean(),
    #              horizontalalignment='center',
    #              verticalalignment='center',
    #              size=6,  # weight='bold',
    #              color='white',
    #              backgroundcolor='black')
    # # Add bounding box
    # box = boxes[label]
    # rect = patches.Rectangle(box[0], box[1][0]-box[0][0], box[1][1]-box[0][1], fill=False)
    # ax.add_patch(rect)

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

