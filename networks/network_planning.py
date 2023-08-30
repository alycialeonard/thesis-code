# Geospatial Design using Delaunay triangulation & Minimum Spanning Tree
# Author Alycia Leonard
# Last updated 2020-02-04

import time
import networkx as nx
import geopandas as gp
import pandas as pd
from math import radians, sin, cos, asin, sqrt
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


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

# A sample of buildings, assumed to be buildings to place in ONE distribution grid
buildings = '/Users/alycia/Documents/SLE Geospatial Design Demo/SLE_KenemaAndBo_Example/SLE_KenemaAndBo/KenemaAndBo_OSMBuildings.geojson'

# Get process start time
start_time = time.time()

print('Loading data...')
# Prepare building data - read as geodataframe
gdf = gp.read_file(buildings)
# Get the number of rows (points) in the GeoDataFrame
no_points = gdf.shape[0]
# Get centroid points of building polygons
gdf['centroid'] = gdf['geometry'].centroid
# Break out lat and long into separate columns of GeoDataFrame
gdf['lon'] = gdf.centroid.x
gdf['lat'] = gdf.centroid.y
# Get lat and long columns from the GeoDataFrame and convert into a numpy array
coords = gdf.drop(['name', 'type', 'code', 'fclass', 'osm_id', 'geometry', 'centroid'], axis=1).to_numpy()

print('Calculating Delaunay triangulation and distance between Delaunay neighbours...')
# Get Delauney triangulation of coordinates
tri = Delaunay(coords)
indices = tri.vertex_neighbor_vertices[0]
indptr = tri.vertex_neighbor_vertices[1]

# Instantiate dictionary to hold neighbors of each point & data-frame to hold distances between neighbours
neighbors = {}
locations = {}
distances = pd.DataFrame(columns=["source", "dest", "distance"])

# Get dictionary of neighbors of all points and a dictionary of locations of all points
for k in range(0, no_points):
    neighbors[k] = indptr[indices[k]:indices[k+1]]
    locations[k] = coords[k][0], coords[k][1]

# Get distances between all Delaunay neighbors
for key, values in neighbors.items():
    for value in values:
        coord_1 = coords[key]
        coord_2 = coords[value]
        dist = haversine_dist_from_coords(coord_1[1], coord_1[0], coord_2[1], coord_2[0])
        distances = distances.append({"source": key, "dest": value, "distance": dist}, ignore_index=True)

print('Creating a graph from this information (edge weight = distance)...')
G = nx.Graph()
for index, row in distances.iterrows():
    G.add_edge(row['source'], row['dest'], weight=row['distance'])

print('Calculating the minimum spanning tree of the graph...')
T = nx.minimum_spanning_tree(G)

print('Minimum spanning tree calculated! Execution time for whole process: ', (time.time() - start_time))

print('Saving tree as shapefiles...')
T_dir = T.to_directed()
nx.write_shp(T_dir, '/Users/alycia/Documents')

edges = T.edges(data=True)
weights = [x[2]['weight'] for x in edges]
total_dist = sum(weights)

print('Number of nodes (buildings) in the graph: ', T.number_of_nodes())
print('Number of edges in the minimum spanning tree: ', T.number_of_edges())
print('Total distance of minimum spanning tree (in km): ', total_dist)

print('Plotting results:')

# Plot home locations
plt.figure()
plt.title('Home locations')
plt.scatter(x=coords[:, 0], y=coords[:, 1], s=20, alpha=1)
plt.xlabel('Longitude ($^\circ$)')
plt.ylabel('Latitude ($^\circ$)')
axes = plt.gca()
axes.set_xlim([min(coords[:, 0]) - 0.001, max(coords[:, 0]) + 0.001])
axes.set_ylim([min(coords[:, 1]) - 0.001, max(coords[:, 1]) + 0.001])

# Plot Delaunay triangulation
plt.figure()
plt.title('Delaunay Triangulation of Homes')
plt.triplot(coords[:, 0], coords[:, 1], tri.simplices)
plt.xlabel('Longitude ($^\circ$)')
plt.ylabel('Latitude ($^\circ$)')
plt.plot(coords[:, 0], coords[:, 1], 'o')
axes = plt.gca()
axes.set_xlim([min(coords[:, 0]) - 0.001, max(coords[:, 0]) + 0.001])
axes.set_ylim([min(coords[:, 1]) - 0.001, max(coords[:, 1]) + 0.001])

# Plot Minimum Spanning Tree made from Delaunay Triangulation
plt.figure()
nx.draw_networkx(T, pos=locations, with_labels=False, node_size=15)
plt.title('Minimum Spanning Tree of Delaunay Graph \n (Edge Weight = Haversine Distance)')
plt.xlabel('Longitude ($^\circ$)')
plt.ylabel('Latitude ($^\circ$)')
axes = plt.gca()
axes.set_xlim([min(coords[:, 0]) - 0.001, max(coords[:, 0]) + 0.001])
axes.set_ylim([min(coords[:, 1]) - 0.001, max(coords[:, 1]) + 0.001])

# Plot relative frequency of edge distances in minimum spanning tree
plt.figure()
plt.hist(weights, bins=100)
plt.yscale("log")
plt.ylabel('Number of edges of this distance')
plt.xlabel('Distance (km)')

plt.show()



