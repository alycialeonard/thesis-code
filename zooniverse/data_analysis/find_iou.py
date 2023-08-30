# Find the intersection over union (IOU), precision, recall, and F1 of annotations.
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
image_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\expert_annotation_subjects\\"

# Read in filenames and get as a list
filenames = pd.read_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\filenames.csv")
filenames_list = filenames['filenames'].to_list()

# Load footprint files as dataframes from CSV
expert = pd.read_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\all_extracted_expert_annotation_footprints_corners_TEST.csv")
#contributor = pd.read_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\contributor_clustered_footprints_minclustersize2_TEST.csv")
contributor = pd.read_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\pttp_annotations.csv")

# Load all annotations
# class_path = "C:\\Users\\alyci/Documents/DPhil/GeoDESA/Zooniverse/DataExports/DataExport_20200922/panoptes/All/shape_extractor_all_prepped.csv"

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
        plt.plot(x_e,y_e, color='green')
        # Plot just the polygon center
        # plt.plot(df_e.loc[j, "x_center"], df_e.loc[j, "y_center"], color='green', marker='o')

    # For each contributor footprint on that image
    for j in range(len(df_c.index)):

        # Make a polygon
        p_c = Polygon([(df_c.loc[j, "x1"], df_c.loc[j, "y1"]), (df_c.loc[j, "x2"], df_c.loc[j, "y2"]),
                     (df_c.loc[j, "x3"], df_c.loc[j, "y3"]), (df_c.loc[j, "x4"], df_c.loc[j, "y4"]),
                     (df_c.loc[j, "x1"], df_c.loc[j, "y1"])])
        # Save the polygon in the dataframe subset
        df_c.at[j, "polygon"] = p_c

        # Plot the polygon
        x_c,y_c = p_c.exterior.xy
        plt.plot(x_c,y_c, color='red')
        # Plot just the polygon center
        # plt.plot(df_c.loc[j, "x1"], df_c.loc[j, "y1"], color='red', marker='o')

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

    # Print counter standings:
    # print('True positives so far: ', TP)
    # print('False positives so far: ', FP)
    # print('False negatives so far: ', FN)

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

