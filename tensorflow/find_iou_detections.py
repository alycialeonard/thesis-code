# Find the intersection over union (IOU), precision, recall, and F1 of CV detections.
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

from shapely import wkt
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Path to images
image_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R0\\expert_annotation_subjects\\"

# Read in filenames and get as a list
filenames = pd.read_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R0\\expert_annotation_filenames.csv")
filenames_list = filenames['filenames'].to_list()

# Load footprint files as dataframes from CSV - contributor is actually detections.
expert = pd.read_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R0\\expert_annotations.csv")
contributor = pd.read_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R_test\\detections\\final_combined_detections_50orhigher.csv")

# Have pandas always print all columns
pd.set_option('display.max_columns', None)

# Set IOU threshold for positive detection
iou_thresh = 0.5

# Create counters for true positive, false positive, false negative
TP, FP, FN = 0, 0, 0

# For each file in the filenames list
for i in filenames_list:

    # Get subset of annotations that correspond to that filename
    df_e = expert.loc[expert['filename'] == i]
    df_c = contributor.loc[contributor['filename'] == i]
    # Reset index of subset to go from 0 to length
    df_e = df_e.reset_index(drop=True)
    df_c = df_c.reset_index(drop=True)
    # Add empty column to hold polygons
    df_e['polygon'] = np.nan
    df_c['polygon'] = np.nan

    # Get path to that image and load it as a numpy array
    # im_path = image_path + i
    # im = np.array(Image.open(im_path), dtype=np.uint8)
    # Start a figure to plot on and add the image
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(im)

    # For each each expert annotation on that image
    for j in range(len(df_e.index)):

        # Make a polygon
        p_e = Polygon([(df_e.loc[j, "x1"], df_e.loc[j, "y1"]), (df_e.loc[j, "x2"], df_e.loc[j, "y2"]),
                     (df_e.loc[j, "x3"], df_e.loc[j, "y3"]), (df_e.loc[j, "x4"], df_e.loc[j, "y4"]),
                     (df_e.loc[j, "x1"], df_e.loc[j, "y1"])])
        # Save the polygon in the dataframe subset
        df_e.at[j, "polygon"] = p_e

        # Plot the polygon
        x_e,y_e = p_e.exterior.xy
        # plt.plot(x_e,y_e, color='green')

    # For each contributor footprint on that image
    for j in range(len(df_c.index)):

        # Make a polygon
        p_c = Polygon([(df_c.loc[j, "xmin_pixel"], df_c.loc[j, "ymin_pixel"]), (df_c.loc[j, "xmin_pixel"], df_c.loc[j, "ymax_pixel"]),
                     (df_c.loc[j, "xmax_pixel"], df_c.loc[j, "ymax_pixel"]), (df_c.loc[j, "xmax_pixel"], df_c.loc[j, "ymin_pixel"]),
                     (df_c.loc[j, "xmin_pixel"], df_c.loc[j, "ymin_pixel"])])
        # Save the polygon in the dataframe subset
        df_c.at[j, "polygon"] = p_c

        # Plot the polygon
        x_c,y_c = p_c.exterior.xy
        # plt.plot(x_c,y_c, color='red')

    # Go through contributor footprints to check for true positives and false positives

    # For each contributor footprint on that image
    for j in range(len(df_c.index)):
        # Get polygon
        p_c = df_c.loc[j, "polygon"]
        # Make a list to hold IOU results
        ious = []
        # For each expert annotation on that image
        for k in range(len(df_e.index)):
            # Get polygon
            p_e = df_e.loc[k, "polygon"]
            # Append intersection of this polygon with contributor polygon to list
            ious.append(p_c.intersection(p_e).area / p_c.union(p_e).area)
        # If any IOU value is greater than or equal to IOU threshold
        if any(y >= iou_thresh for y in ious):
            # This is a true positive; increment the TP counter
            TP = TP+1
        # Otherwise
        else:
            # This is a false positive; increment the FP counter
            FP = FP+1

    # Go through expert annotations to check for false negatives

    # For each expert annotation on that image
    for j in range(len(df_e.index)):
        # Get polygon
        p_e = df_e.loc[j, "polygon"]
        # Make a list to hold IOU results
        ious = []
        # For each contributor footprint on that image
        for k in range(len(df_c.index)):
            # Get polygon
            p_c = df_c.loc[k, "polygon"]
            # Append intersection of this polygon with contributor polygon to list
            ious.append(p_e.intersection(p_c).area / p_e.union(p_c).area)
        # If there is no IOU value greater than or equal to the IOU threshold
        if not any(y >= iou_thresh for y in ious):
            # This is a false negative; increment the FN counter.
            FN = FN+1

    # Finalize plot settings and show
    # ax.set_aspect('equal', adjustable='box')
    # plt.xlim(0,512)
    # plt.ylim(0,512)
    # plt.gca().invert_yaxis()
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    # plt.show()

# Get precision, recall, and F1
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1 = (precision * recall)/((precision + recall)/2)

# Print these
print("TP: ", TP)
print("FP: ", FP)
print("FN: ", FN)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)

