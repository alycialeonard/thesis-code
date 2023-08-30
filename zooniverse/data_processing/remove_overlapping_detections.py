# Remove any detections which overlap more than 0.75 IoU (i.e. likely to be duplicates)
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

from shapely.geometry import Polygon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to images
image_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R0\\expert_annotation_subjects\\"

# Read in filenames and get as a list
filenames = pd.read_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R0\\expert_annotation_filenames.csv")
filenames_list = filenames['filenames'].to_list()

# Load footprint files as dataframes from CSV - contributor is actually detections
contributor = pd.read_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R0\\detections\\final_combined_detections_50orhigher.csv")

# Have pandas always print all columns
pd.set_option('display.max_columns', None)

# Set IOU threshold for likely duplicate (previously positive detection)
iou_thresh = 0.3

# Create counters for dups and keeps
dup, keep = 0, 0

# Create empty dataframe to hold non-duplicates
df = pd.DataFrame()

# For each file in the filenames list
for i in filenames_list:

    print("\nCHECKING DETECTIONS FOR IMAGE: " + str(i))

    # Get subset of annotations that correspond to that filename
    df_c = contributor.loc[contributor['filename'] == i]
    # Reset index of subset to go from 0 to length
    df_c = df_c.reset_index(drop=True)
    # Add empty column to hold polygons
    df_c['polygon'] = np.nan

    # Make a list full of zeros to hold IOU results
    # ious = [0] * len(df_c.index)
    # print(ious)

    # Get path to that image and load it as a numpy array
    im_path = image_path + i
    im = np.array(Image.open(im_path), dtype=np.uint8)
    # Start a figure to plot on and add the image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)

    # Make a polygon in the dataframe for each detection on that image
    for j in range(len(df_c.index)):
        p = Polygon([(df_c.loc[j, "xmin_pixel"], df_c.loc[j, "ymin_pixel"]),
                    (df_c.loc[j, "xmin_pixel"], df_c.loc[j, "ymax_pixel"]),
                    (df_c.loc[j, "xmax_pixel"], df_c.loc[j, "ymax_pixel"]),
                    (df_c.loc[j, "xmax_pixel"], df_c.loc[j, "ymin_pixel"]),
                    (df_c.loc[j, "xmin_pixel"], df_c.loc[j, "ymin_pixel"])])
        # Save the polygon in the dataframe subset
        df_c.at[j, "polygon"] = p

    # Go through the detections to check for duplicates

    # For each detection on that image
    for j in df_c.index.values.tolist(): # j in range(len(df_c.index)):

        # Check to make sure this row hasn't been removed since the start
        if j not in df_c.index.values.tolist():
            print("Detection " + str(j) + " was already removed as a duplicate!")
            continue

        else:
            print("Checking detection " + str(j) + "...")

        # For each other detection on that image
        for k in df_c.index.values.tolist():  # range(len(df_c.index)): #for k, row in df.iterrows():

            print("-- Checking against detection " + str(k) + "...")

            # if k is the same as j (i.e. same polygon), append 0 and continue. Otherwise, proceed.
            if j == k:
                #ious.append(0)
                print("---- Skipping " + str(k) + " = " + str(j) + ".")
                continue

            else:
                # Get IoU of detection j with detection k
                p = df_c.loc[j, "polygon"]
                p_test = df_c.loc[k, "polygon"]

                # ious[k] = p.intersection(p_test).area / p.union(p_test).area
                iou = p.intersection(p_test).area / p.union(p_test).area
                # print("IoU: " + str(iou))
                if iou >= iou_thresh:  # #if any(y >= iou_thresh for y in ious):

                    # This is a duplicate; increment the dup counter and remove this detection
                    print("---- Duplicate found! IoU = " + str(iou) + " for detection " + str(k))
                    dup = dup + 1
                    df_c.drop([k], inplace=True)  # ious.index(y)
                    #print(df_c)

                # Otherwise, This polygon is fine, increment the keep counter.
                else:
                    keep = keep + 1

        #print("IoU values: ")
        #print(ious)

        # If any IOU value is greater than or equal to IOU threshold
        #for y in ious:

        #    if y >= iou_thresh:  # #if any(y >= iou_thresh for y in ious):

                # This is a duplicate; increment the dup counter and remove this detection
        #        print("Duplicate found! IoU = " + str(y) + " for detection " + str(ious.index(y)))
        #        dup = dup + 1
        #        df_c.drop([ious.index(y)], inplace=True)  # ious.index(y)
        #        print(df_c)

            # Otherwise, This polygon is fine, increment the keep counter.
        #    else:

         #       keep = keep+1

    # Append the df_c without duplicates to the dataframe
    df = df.append(df_c)

    # Plot all valid polygons on the image
    for j in df_c.index.values.tolist():  # range(len(df_c.index)):

        # Plot the polygon
        p = df_c.loc[j, "polygon"]
        x,y = p.exterior.xy
        plt.plot(x,y, color='red')

    # Finalize plot settings and show
    ax.set_aspect('equal', adjustable='box')
    plt.xlim(0,512)
    plt.ylim(0,512)
    plt.gca().invert_yaxis()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R0\\detections\\visualisations_dupsremoved30\\" + str(i) + "_nodups.png")
    plt.close()

# Save the dataframe containing annotations with no duplicates
df.to_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R0\\detections\\final_combined_detections_50orhigher_dupsremoved30.csv")

# Print the counts of dups and keeps
print("dups: ", dup)
print("keeps: ", keep)
