# Converts from x, y, angle, width to all four corners in annotations
# Also appends subject data and georeferencing data for funsies
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk


import pandas as pd
# import numpy as np
from pathlib import Path
# import json
from ast import literal_eval
import math


def get_filename(meta):
    return literal_eval(meta)["Pansharpened"]


# File paths
data_path = Path("C:/Users/alyci/Documents/DPhil/GeoDESA") #/Zooniverse/DataExports")

tiles_path = data_path / "PS_Tiles_Georeference.csv" # "DataExport_20200922/georeferencing/PS_Tiles_Georeference.csv"
subjects_path = data_path / "power-to-the-people-subjects.csv" #"DataExport_20201115/power-to-the-people-subjects-live.csv"
footprints_path = data_path / "annotations.csv" # "DataExport_20200922/panoptes/All/min_cluster_size_10/shape_reducer_hdbscan_all_minclustersize_10_footprints.csv"

# Open the tiles, subjects, and footprints CSVs in pandas
tiles = pd.read_csv(tiles_path)
subjects = pd.read_csv(subjects_path)
footprints = pd.read_csv(footprints_path)
#merged = pd.read_csv(footprints_path)

# Join footprints and subject data on subject_id
merged = pd.merge(left=footprints, right=subjects, on='subject_id', how='left')
# Extract filename into separate column
merged['filename'] = merged["metadata"].apply(get_filename)
# Join geo-referencing data for the subject on the filename
merged = pd.merge(left=merged, right=tiles, on='filename', how='left')

# Get all four corners of each box, rotated to angle listed

# First, get unrotated corners
merged['x1_unrotated'] = merged['x']
merged['y1_unrotated'] = merged['y']
merged['x2_unrotated'] = merged.apply(lambda row: row.x + row.width, axis=1)
merged['y2_unrotated'] = merged['y']
merged['x3_unrotated'] = merged.apply(lambda row: row.x + row.width, axis=1)
merged['y3_unrotated'] = merged.apply(lambda row: row.y + row.height, axis=1)
merged['x4_unrotated'] = merged['x']
merged['y4_unrotated'] = merged.apply(lambda row: row.y + row.height, axis=1)

# Then, get box center
merged['x_center'] = merged.apply(lambda row: row.x + (row.width/2), axis=1)
merged['y_center'] = merged.apply(lambda row: row.y + (row.height/2), axis=1)

# Then, rotate each point
merged['x1'] = merged.apply(lambda row: ((row.x1_unrotated - row.x_center)*math.cos(-row.angle*math.pi/180) + (row.y1_unrotated - row.y_center)*math.sin(-row.angle*math.pi/180)) + row.x_center, axis=1)
merged['y1'] = merged.apply(lambda row: (-(row.x1_unrotated - row.x_center)*math.sin(-row.angle*math.pi/180) + (row.y1_unrotated - row.y_center)*math.cos(-row.angle*math.pi/180)) + row.y_center, axis=1)
merged['x2'] = merged.apply(lambda row: ((row.x2_unrotated - row.x_center)*math.cos(-row.angle*math.pi/180) + (row.y2_unrotated - row.y_center)*math.sin(-row.angle*math.pi/180)) + row.x_center, axis=1)
merged['y2'] = merged.apply(lambda row: (-(row.x2_unrotated - row.x_center)*math.sin(-row.angle*math.pi/180) + (row.y2_unrotated - row.y_center)*math.cos(-row.angle*math.pi/180)) + row.y_center, axis=1)
merged['x3'] = merged.apply(lambda row: ((row.x3_unrotated - row.x_center)*math.cos(-row.angle*math.pi/180) + (row.y3_unrotated - row.y_center)*math.sin(-row.angle*math.pi/180)) + row.x_center, axis=1)
merged['y3'] = merged.apply(lambda row: (-(row.x3_unrotated - row.x_center)*math.sin(-row.angle*math.pi/180) + (row.y3_unrotated - row.y_center)*math.cos(-row.angle*math.pi/180)) + row.y_center, axis=1)
merged['x4'] = merged.apply(lambda row: ((row.x4_unrotated - row.x_center)*math.cos(-row.angle*math.pi/180) + (row.y4_unrotated - row.y_center)*math.sin(-row.angle*math.pi/180)) + row.x_center, axis=1)
merged['y4'] = merged.apply(lambda row: (-(row.x4_unrotated - row.x_center)*math.sin(-row.angle*math.pi/180) + (row.y4_unrotated - row.y_center)*math.cos(-row.angle*math.pi/180)) + row.y_center, axis=1)

# Get box coordinates
merged['x1_coord'] = merged.apply(lambda row: row.tile_min_x + (row.x1/row.tile_pixel_width) * row.tile_geo_width, axis=1)
merged['y1_coord'] = merged.apply(lambda row: row.tile_max_y - (row.y1/row.tile_pixel_height) * row.tile_geo_height, axis=1)
merged['x2_coord'] = merged.apply(lambda row: row.tile_min_x + (row.x2/row.tile_pixel_width) * row.tile_geo_width, axis=1)
merged['y2_coord'] = merged.apply(lambda row: row.tile_max_y - (row.y2/row.tile_pixel_height) * row.tile_geo_height, axis=1)
merged['x3_coord'] = merged.apply(lambda row: row.tile_min_x + (row.x3/row.tile_pixel_width) * row.tile_geo_width, axis=1)
merged['y3_coord'] = merged.apply(lambda row: row.tile_max_y - (row.y3/row.tile_pixel_height) * row.tile_geo_height, axis=1)
merged['x4_coord'] = merged.apply(lambda row: row.tile_min_x + (row.x4/row.tile_pixel_width) * row.tile_geo_width, axis=1)
merged['y4_coord'] = merged.apply(lambda row: row.tile_max_y - (row.y4/row.tile_pixel_height) * row.tile_geo_height, axis=1)

# Export result
merged.to_csv(data_path / "annotations_georeferenced.csv")

# OLD AND WRONG TRIG FOR ROTATION ABOUT CORNER
#merged['x2'] = merged.apply(lambda row: row.x + row.width*math.cos(row.angle*math.pi/180), axis=1)
#merged['y2'] = merged.apply(lambda row: row.y + row.width*math.sin(row.angle*math.pi/180), axis=1)
#merged['x3'] = merged.apply(lambda row: row.x + row.width*math.cos(row.angle*math.pi/180) - row.height*math.sin(row.angle*math.pi/180), axis=1)
#merged['y3'] = merged.apply(lambda row: row.y + row.width*math.sin(row.angle*math.pi/180) + row.height*math.cos(row.angle*math.pi/180), axis=1)
#merged['x4'] = merged.apply(lambda row: row.x - row.height*math.sin(row.angle*math.pi/180), axis=1)
#merged['y4'] = merged.apply(lambda row: row.y + row.height*math.cos(row.angle*math.pi/180), axis=1)