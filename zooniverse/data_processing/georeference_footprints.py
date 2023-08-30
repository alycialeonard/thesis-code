# This script takes a combined georeferencing list for all image tiles,
# the subjects export from the Power to the People platform (restricted to live/relevant workflows)
# and the aggregated/clustered building footprints generated with create_aggregated_annotation_csv.
# It retrieves the georeferencing data relevant to each building footprint
# and associates geographic coordinates with each building footprint.
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
import numpy as np
from pathlib import Path
import json
from ast import literal_eval
import math

def get_filename(meta):
    return literal_eval(meta)["Pansharpened"]


# File paths
data_path = Path("C:/Users/alyci/Documents/DPhil/GeoDESA/Zooniverse/DataExports")
tiles_path = data_path / "DataExport_20200922/georeferencing/PS_Tiles_Georeference.csv"
# subjects_path = data_path / "DataExport_20201115/power-to-the-people-subjects-live.csv"
# footprints_path = data_path / "DataExport_20200922/panoptes/All/min_cluster_size_3/shape_reducer_hdbscan_all_minclustersize_3_footprints.csv"
footprints_path = data_path / "ExpertAnnotation_20210713/finals_for_comparison/pttp_annotations.csv"

# Open the tiles, subjects, and footprints CSVs in pandas
tiles = pd.read_csv(tiles_path)
# subjects = pd.read_csv(subjects_path)
footprints = pd.read_csv(footprints_path)

# Drop the workflow column from the footprints - it's not right and we bring workflow in from subject data later
# footprints = footprints.drop(columns=['workflow_id'])

# Join footprints and subjects data on subject_id
# merged = pd.merge(left=footprints, right=subjects, on='subject_id', how='left') # ['subject_id', 'workflow_id']
# Extract filename into separate column
# merged['filename'] = merged["metadata"].apply(get_filename)
# Join georeferencing data for the subject on the filename
merged = pd.merge(left=footprints, right=tiles, on='filename', how='left')

# Get all four corners of each box, rotated to angle listed - NO ROTATION TEST commented out
merged['x1'] = merged['x']
merged['y1'] = merged['y']
#merged['x2'] = merged['x'] + merged['width']
#merged['y2'] = merged['y']
#merged['x3'] = merged['x'] + merged['width']
#merged['y3'] = merged['y'] + merged['height']
#merged['x4'] = merged['x']
#merged['y4'] = merged['y'] + merged['height']

merged['x2'] = merged.apply(lambda row: row.x + row.width*math.cos(row.angle*math.pi/180), axis=1)
merged['y2'] = merged.apply(lambda row: row.y + row.width*math.sin(row.angle*math.pi/180), axis=1)
merged['x3'] = merged.apply(lambda row: row.x + row.width*math.cos(row.angle*math.pi/180) - row.height*math.sin(row.angle*math.pi/180), axis=1)
merged['y3'] = merged.apply(lambda row: row.y + row.width*math.sin(row.angle*math.pi/180) + row.height*math.cos(row.angle*math.pi/180), axis=1)
merged['x4'] = merged.apply(lambda row: row.x - row.height*math.sin(row.angle*math.pi/180), axis=1)
merged['y4'] = merged.apply(lambda row: row.y + row.height*math.cos(row.angle*math.pi/180), axis=1)

# Get bottom left box coordinates - THIS IS WRONG. See corner_coords_from_annotations.py
merged['x1_coord'] = merged.apply(lambda row: row.tile_min_x + (row.x1/row.tile_pixel_width) * row.tile_geo_width, axis=1)
#merged['y1_coord'] = merged.apply(lambda row: row.tile_min_y + (row.y1/row.tile_pixel_height) * row.tile_geo_height, axis=1)
merged['y1_coord'] = merged.apply(lambda row: row.tile_max_y - (row.y1/row.tile_pixel_height) * row.tile_geo_height, axis=1)
merged['x2_coord'] = merged.apply(lambda row: row.tile_min_x + (row.x2/row.tile_pixel_width) * row.tile_geo_width, axis=1)
#merged['y2_coord'] = merged.apply(lambda row: row.tile_min_y + (row.y2/row.tile_pixel_height) * row.tile_geo_height, axis=1)
merged['y2_coord'] = merged.apply(lambda row: row.tile_max_y - (row.y2/row.tile_pixel_height) * row.tile_geo_height, axis=1)
merged['x3_coord'] = merged.apply(lambda row: row.tile_min_x + (row.x3/row.tile_pixel_width) * row.tile_geo_width, axis=1)
#merged['y3_coord'] = merged.apply(lambda row: row.tile_min_y + (row.y3/row.tile_pixel_height) * row.tile_geo_height, axis=1)
merged['y3_coord'] = merged.apply(lambda row: row.tile_max_y - (row.y3/row.tile_pixel_height) * row.tile_geo_height, axis=1)
merged['x4_coord'] = merged.apply(lambda row: row.tile_min_x + (row.x4/row.tile_pixel_width) * row.tile_geo_width, axis=1)
#merged['y4_coord'] = merged.apply(lambda row: row.tile_min_y + (row.y4/row.tile_pixel_height) * row.tile_geo_height, axis=1)
merged['y4_coord'] = merged.apply(lambda row: row.tile_max_y - (row.y4/row.tile_pixel_height) * row.tile_geo_height, axis=1)

#Export result
merged.to_csv(data_path / "ExpertAnnotation_20210713/finals_for_comparison/pttp_annotations.csv_2.csv")



#x1_coord = tile_min_x + (x/tile_pixel_width)*(tile_geo_width)
#y1_coord = tile_min_y + (y/tile_pixel_height)*(tile_geo_height)

#x2 = x + width*math.cos(angle) + height*math.sin(angle)
#y2 = y - width*math.sin(angle) + height*math.cos(angle)

#merged['x1_coord'] = merged.apply(get_x1, axis=1)
#def get_x1(row):
#    return row['tile_min_x'] + (row['x'] / row['tile_pixel_width']) * row['tile_geo_width']

# Get top right corner pixel value for the box based on angle
#merged['x2'] = merged.apply(lambda row: row.x + row.width*math.cos(row.angle*math.pi/180) - row.height*math.sin(row.angle), axis=1)
#merged['y2'] = merged.apply(lambda row: row.y + row.width*math.sin(row.angle*math.pi/180) + row.height*math.cos(row.angle), axis=1)

# Get top right box coordinates
#x2_coord = tile_min_x + (x2/tile_pixel_width)*(tile_geo_width)
#y2_coord = tile_min_y + (y2/tile_pixel_height)*(tile_geo_height)

# Attach filename from subjects to rows of footprints on subject_id

# Add column to footprints called x_coordinates


# For each footprint
    # Get the image tile it comes from
    # Get the coordinates of the corners of the image
    # Get the proportion of the image in each direction which is at each coordinate

# return literal_eval(meta['Pansharpened'])
# return json.loads(metadata_str).get('Pansharpened')

# print(merged.head())
# print(len(merged.index))