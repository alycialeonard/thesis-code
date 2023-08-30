# Clustering and analysis GUI for intended Energy Policy submission
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk
# Date: 2019-09-30

import sys
import os
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from osgeo import gdal
import pandas as pd
import geopandas as gp
from rasterstats import zonal_stats
from shapely.geometry import shape, Point, Polygon, mapping, box
import json
from skimage.filters import threshold_otsu
import sklearn.cluster as cluster
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import UnivariateSpline
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
from osgeo.gdalconst import *
from tkinter import *
from tkinter import filedialog
import fiona
import plotly.graph_objects as go
import plotly
import plotly.express as px


# --------------- PARAMETERS --------------- #

DBSCAN_MINPTS = 5
K = 5


# --------------- CONSTANTS --------------- #

R_EARTH_KM = 6371


# --------------- GUI LOADER ---------------#

# Create empty variables for data paths
#
# building_data_path: OpenStreetMap building footprints data, trimmed to region of interest,
#   downloaded from http://download.geofabrik.de/africa.html as either Shapefile or GeoJSON
# nightlight_data: NASA Black Marble night lights data at 15 arc-second (i.e. 0.00416667 deg) resolution,
#   trimmed to bounding box of country containing region of interest,
#   downloaded from https://earthobservatory.nasa.gov/features/NightLights
# wind_data_path: wind power density in W/m^2 at 9 arc-second (i.e. 0.0025 deg) resolution
#   downloaded from https://globalwindatlas.info/
# pv_data_path: photovoltaic power potential in kWh/kWp at 30 arc-second (i.e. 0.00833333 deg) resolution
#   downloaded from https://globalsolaratlas.info/
# country_path: Vector file of country boundaries produced by UN OCHA
#   downloaded from https://data.humdata.org/ - search for boundaries geodata produced by OCHA
# region_data: Vector file of all Admin level 1 boundaries
#   downloaded from https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/
#   https://gadm.org/index.html

building_data_path, nightlight_data_path, wind_data_path, pv_data_path, country_path, roi_polygon_path = '', '', '', '', '', ''
# region_data_path ''

# Create variable to hold whether coordinates were provided (1) or not (0) for region of interest, and lists to hold coords
bounding_box_provided = False
polygon_provided = False
lon_points = []
lat_points = []

# Create empty variable to hold identifier for this run and region level
identifier = ''
region_level = ''

# Create empty variable to hold directory where results are saved
save_directory = ''

# Set up window
root = Tk()
root.title('Energy Clustering & Characterisation')

# Define button functions


def open_buildings():
    root.filename = filedialog.askopenfilename(initialdir="C:\\Users\\alyci\\Documents\\DPhil\\Papers\\Energy Policy", title="Select a File")
    global building_data_path
    building_data_path = root.filename


def open_nightlight():
    root.filename = filedialog.askopenfilename(initialdir="C:\\Users\\alyci\\Documents\\DPhil\\Papers\\Energy Policy", title="Select a File")
    global nightlight_data_path
    nightlight_data_path = root.filename


def open_wind():
    root.filename = filedialog.askopenfilename(initialdir="C:\\Users\\alyci\\Documents\\DPhil\\Papers\\Energy Policy", title="Select a File")
    global wind_data_path
    wind_data_path = root.filename


def open_pv():
    root.filename = filedialog.askopenfilename(initialdir="C:\\Users\\alyci\\Documents\\DPhil\\Papers\\Energy Policy", title="Select a File")
    global pv_data_path
    pv_data_path = root.filename


def open_country():
    root.filename = filedialog.askopenfilename(initialdir="C:\\Users\\alyci\\Documents\\DPhil\\Papers\\Energy Policy", title="Select a File")
    global country_path
    country_path = root.filename

def open_roi():
    root.filename = filedialog.askopenfilename(initialdir="C:\\Users\\alyci\\Documents\\DPhil\\Papers\\Energy Policy", title="Select a File")
    global roi_polygon_path
    roi_polygon_path = root.filename

# def open_region():
#     root.filename = filedialog.askopenfilename(initialdir="/", title="Select a File")
#     global region_data_path
#     region_data_path = root.filename


def save_directory_choice():
    root.filename = filedialog.askdirectory(initialdir="C:\\Users\\alyci\\Documents\\DPhil\\Papers\\Energy Policy", title="Save Location")
    global save_directory
    save_directory = root.filename


def close_window():
    global identifier
    identifier = identifier_entry.get()
    global bounding_box_provided
    bounding_box_provided = var1.get()
    global polygon_provided
    polygon_provided = var2.get()
    global lat_points
    lat_points.append(float(lat1_entry.get()))
    lat_points.append(float(lat2_entry.get()))
    global lon_points
    lon_points.append(float(lon1_entry.get()))
    lon_points.append(float(lon2_entry.get()))
#    global region_level
#    region_level = region_level_entry.get()
    root.destroy()



# Create and pack widgets
identifier_entry_label = Label(root, text="Enter identifier text to prepend to output file names for this run:").pack()
identifier_entry = Entry(root, width=50)
identifier_entry.pack()
tickbox_label = Label(root, text="Use bounding box or imported shapefile to define region of interest?").pack()
var1 = BooleanVar()
bounding_box_tick = Checkbutton(root, text="Bounding box", variable=var1).pack()
var2 = BooleanVar()
polygon_tick = Checkbutton(root, text="Polygon", variable=var2).pack()
bounding_box_entry_label = Label(root, text="Enter bounding box coordinates if used:").pack()
lat1_entry = Label(text="Minimum Latitude:").pack()
lat1_entry = Entry(root, width=30)
lat1_entry.pack()
lat2_label = Label(text="Maximum Latitude:").pack()
lat2_entry = Entry(root, width=30)
lat2_entry.pack()
lon1_label = Label(text="Minimum Longitude:").pack()
lon1_entry = Entry(root, width=30)
lon1_entry.pack()
lon2_label = Label(text="Maximum Longitude:").pack()
lon2_entry = Entry(root, width=30)
lon2_entry.pack()
polygon_label = Label(text="Or, load region of interest polygon:").pack()
polygon_button = Button(root, text="Load Region of Interest Shapefile", command=open_roi).pack()
polygon_label = Label(text="LOAD ALL NECESSARY DATA:").pack()
building_button = Button(root, text="Load Building Data", command=open_buildings).pack()
nightlight_button = Button(root, text="Load Night Light Data", command=open_nightlight).pack()
wind_button = Button(root, text="Load Wind Power Density", command=open_wind).pack()
pv_button = Button(root, text="Load PV Power Potential", command=open_pv).pack()
country_button = Button(root, text="Load Country Borders", command=open_country).pack()
# region_button = Button(root, text="Load Regional Borders", command=open_region).pack()
# region_label = Label(root, text="Enter region level being used (i.e. 1, 2, 3):").pack()
# region_level_entry = Entry(root, width=5).pack()
save_directory_button = Button(root, text="Choose Directory Where Results Will Be Saved", command=save_directory_choice).pack()

# Create ready button
done_button = Button(root, text="Ready to go!!", command=close_window, bg='red').pack()

# Run window until user clicks "All data loaded!" button
root.mainloop()

# --------------- FILEPATHS --------------- #

# Filepaths to country and regional boundary files (GeoJSON)
# produced by United Nations Office for the Coordination of Humanitarian Affairs
# downloaded from https://data.humdata.org/dataset
# For SLE, regions are districts; for KEN, regions are counties; for UGA, regions are districts.
# For SLE, region_property is 'admin2Name'; For UGA, "ADM1_EN"; For KEN, "ADM1_EN"

# regions = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\country_boundaries\\sle_admbnda_adm2_1m_gov_ocha\\sle_admbnda_adm2_1m_gov_ocha_20161017.geojson'
# region_property = "admin2Name"

# Filepaths to save final results
clustering_results = os.path.join(save_directory, identifier + '_clustering_results.geojson')
analysis_results_csv = os.path.join(save_directory, identifier + '_analysis_results.csv')
analysis_results = os.path.join(save_directory, identifier + '_analysis_results.geojson')
console_output = os.path.join(save_directory, identifier + '_console_output.txt')
knee_graph = os.path.join(save_directory, identifier + '_knee_graph.png')
cluster_graph = os.path.join(save_directory, identifier + '_cluster_graph.png')
interactive_graph_pv = os.path.join(save_directory, identifier + '_interactive_graph_pv.html')
interactive_graph_wind = os.path.join(save_directory, identifier + '_interactive_graph_wind.html')

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


# Calculate luminance from an RGB geotiff image based on ITU BT.709 https://www.itu.int/rec/R-REC-BT.709
# Takes the filepath of the original image and destination filepath for output image (including extension)
def luminance_from_RGB(original_filepath, output_filepath):
    # Load data and get geotransform
    gdata = gdal.Open(original_filepath)
    gt = gdata.GetGeoTransform()
    data = gdata.ReadAsArray().astype(np.float)
    # Get luminance matrix for night light data
    data_luminance = 0.2126 * data[0] + 0.7152 * data[1] + 0.0722 * data[2]
    # Register all of the GDAL drivers
    gdal.AllRegister()
    # Get number of rows and columns in the data
    gdata_rows = gdata.RasterYSize
    gdata_cols = gdata.RasterXSize
    # Create the output image
    driver = gdata.GetDriver()
    # Print driver
    outDs = driver.Create(output_filepath, gdata_cols, gdata_rows, 1, GDT_Int32)
    if outDs is None:
        print('Could not create output file')
        sys.exit(1)
    # Say where to write the output data, and what to write
    outBand = outDs.GetRasterBand(1)
    outData = data_luminance
    # Write the data
    outBand.WriteArray(outData, 0, 0)
    # Flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    outBand.SetNoDataValue(-99)
    # Georeference the image and set the projection
    outDs.SetGeoTransform(gt)
    outDs.SetProjection(gdata.GetProjection())
    del outData


# --------------- PROCESS --------------- #

# Get process start time
start_time = datetime.now()

# Set stdout to print to output file
#sys.stdout = open(console_output, 'w')

# Import building data as GeoDataFrame
print('Reading building footprint data into a GeoDataFrame...')
OSM_data_full = gp.read_file(building_data_path)
print('GeoDataFrame is ready!')

print(lat_points)
print(lon_points)

mask = 0
# Get polygon for region of interest
if bounding_box_provided and not polygon_provided:
    # mask = Polygon(zip(lon_points, lat_points))
    mask = box(lon_points[0], lat_points[0], lon_points[1], lat_points[1])
elif polygon_provided and not bounding_box_provided:
    mask = fiona.open(roi_polygon_path)
else:
    print("Too many region of interest definitions provided!")
    exit()

# Trim building data to the region of interest
OSM_data_full['roi'] = OSM_data_full.intersects(mask)
OSM_data = gp.GeoDataFrame(OSM_data_full.loc[OSM_data_full['roi'] == True])
# Get rid of ROI column
OSM_data = OSM_data.drop(['roi'], axis=1)

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
pv = {}
luminance = {}
bright = {}
boxes = {}
area = {}
building_count = {}
region = {}

# Calculate luminance from night light data and save in results directory
luminance_data_path = os.path.join(save_directory, identifier + '_luminance_data.tif')
luminance_from_RGB(nightlight_data_path, luminance_data_path)

print("Reading in complementary data (wind, PV, night-time luminance, regional boundaries)...")
# Read in wind data
wind_gdata = gdal.Open(wind_data_path)
wind_gt = wind_gdata.GetGeoTransform()
wind_data = wind_gdata.ReadAsArray().astype(np.float)
# Read in PV data
pv_gdata = gdal.Open(pv_data_path)
pv_gt = pv_gdata.GetGeoTransform()
pv_data = pv_gdata.ReadAsArray().astype(np.float)
# Read in luminance data
luminance_gdata = gdal.Open(luminance_data_path)
luminance_gt = luminance_gdata.GetGeoTransform()
luminance_data = luminance_gdata.ReadAsArray().astype(np.float)
luminance_data[luminance_data == -99] = 0
# Read in the regions vector file
# js = json.load(open(regions))
# print("Data loaded!")

# Read in the GDAM regions vector file
# region_data = fiona.open(region_data_path)
# Region data is multipolygon. Using "next", get one polygon of multipolygon
# region = region_data.next()


print("Calculating statistics for raster data...")
# Get stats for PV raster
pv_stats = zonal_stats(country_path, pv_data_path, stats=['min', 'max', 'mean', 'median'])
print('PV stats: ', pv_stats)
# Get stats for wind raster
wind_stats = zonal_stats(country_path, wind_data_path, stats=['min', 'max', 'mean', 'median'])
print('Wind stats: ', wind_stats)
# Get stats for luminance raster:
luminance_stats = zonal_stats(country_path, luminance_data_path, stats=['min', 'max', 'mean', 'median'])
print('Luminance stats: ', luminance_stats)
# Get Otsu threshold of luminance
otsu_thresh = threshold_otsu(luminance_data)
print('Otsu threshold for luminance raster: ', otsu_thresh)

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
    # Get the pv value at the mean point of the cluster
    x = int((mean_coord[cluster][0] - pv_gt[0])/pv_gt[1])
    y = int((mean_coord[cluster][1] - pv_gt[3])/pv_gt[5])
    pv[cluster] = pv_data[y, x]
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
#    mean_coord_point = Point(mean_coord[cluster])
        # for all polygons
        #     if mean_coord_point.within(shape(region['geometry'])):
        #         save the name of the polygon

# Close all datasets
wind_gdata = None
pv_gdata = None
luminance_gdata = None

print('Data gathering complete!')

# Make a dataframe from all this data
cluster_summary_df = pd.DataFrame.from_dict(mean_coord, orient='index')
# cluster_summary_df["region"] = pd.Series(region)
cluster_summary_df["building_count"] = pd.Series(building_count)
cluster_summary_df["bounding_box_area"] = pd.Series(area)
cluster_summary_df["wind"] = pd.Series(wind)
cluster_summary_df["pv_out"] = pd.Series(pv)
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

# PLOT 1

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

# PLOT 2

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
                    s=20, color='black', alpha=1)
    else:
        plt.scatter(x=OSM_data.loc[OSM_data['cluster_index_1'] == label, 'lon'],
                    y=OSM_data.loc[OSM_data['cluster_index_1'] == label, 'lat'],
                    s=20, alpha=1)

# plt.title('Clusters (DBSCAN: eps = knee point, minpts = 5)')
plt.xlabel('Longitude ($^\circ$)', fontsize=16)
plt.ylabel('Latitude ($^\circ$)', fontsize=16)
ax.set_aspect('equal', 'box')
plt.xticks(size=16)
plt.yticks(size=16)
fig.set_size_inches(16, 16)
plt.show(block=False)
plt.savefig(cluster_graph)

# PLOT 3

# Making interactive HTML cluster visualisation
print("Creating interactive visualisations...")

fig = go.Figure()

fig.add_trace(go.Scattermapbox(
        lon=cluster_summary_gdf["lon"],
        lat=cluster_summary_gdf["lat"],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=cluster_summary_gdf["building_count"],
            sizemode="area",
            sizemin=5,
            # sizeref=100 - use this if I decide to set sizemode back to diameter
            color=cluster_summary_gdf["pv_out"],
            colorscale="ylorrd",
            showscale=True,
            opacity=0.4,
            colorbar=dict(
                title=dict(
                    text="PV Potential (kWh/kWp)"
                )
            )
        ),
        text=cluster_summary_gdf['building_count'].astype(str) + ' buildings',
        hoverinfo='text'
    ))

fig.update_layout(
    title='<b>Electrification Clusters - PV Potential</b><br>Marker area proportional to # of buildings in the cluster',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox=go.layout.Mapbox(
        style='open-street-map',
        center=dict(
            lon=cluster_summary_gdf["lon"][1],
            lat=cluster_summary_gdf["lat"][1]
        ),
        zoom=10
    ),
)

plotly.offline.plot(fig, filename=interactive_graph_pv)


# PLOT 4

fig2 = go.Figure()

fig2.add_trace(go.Scattermapbox(
        lon=cluster_summary_gdf["lon"],
        lat=cluster_summary_gdf["lat"],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=cluster_summary_gdf["building_count"],
            sizemode="area",
            sizemin=5,
            # sizeref=100 - use this if I decide to set sizemode back to diameter
            color=cluster_summary_gdf["wind"],
            colorscale="Blues",
            showscale=True,
            opacity=0.4,
            colorbar=dict(
                title=dict(
                    text="Wind Power Density (W/m^2)"
                )
            )
        ),
        text=cluster_summary_gdf['building_count'].astype(str) + ' buildings',
        hoverinfo='text'
    ))

fig2.update_layout(
    title='<b>Electrification Clusters - Wind Potential </b><br>Marker area proportional to # of buildings in the cluster',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox=go.layout.Mapbox(
        style='open-street-map',
        center=dict(
            lon=cluster_summary_gdf["lon"][1],
            lat=cluster_summary_gdf["lat"][1]
        ),
        zoom=10
    ),
)

plotly.offline.plot(fig2, filename=interactive_graph_wind)


print("Plots closed!")

