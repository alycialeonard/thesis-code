# Plots TF training data on relevant images
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.u

from shapely import wkt
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Path to images
image_path = "D:\\fasterrcnn_20210804\\test\\"

# Read in filenames and get as a list
filenames = pd.read_csv("D:\\fasterrcnn_20210804\\test_images_20210804.csv")
filenames_list = filenames['filenames'].to_list()

# Load footprint files as dataframes from CSV
df = pd.read_csv("D:\\fasterrcnn_20210804\\test_labels_20210804.csv")

# Have pandas always print all columns
pd.set_option('display.max_columns', None)

# For each file in the filenames list
for i in filenames_list:

    # Get subset of annotations that correspond to that filename
    df_e = df.loc[df['filename'] == i]
    # Reset index of subset to go from 0 to length
    df_e = df_e.reset_index(drop=True)
    # Add empty column to hold polygons
    df_e['polygon'] = np.nan

    # Get path to that image and load it as a numpy array
    im_path = image_path + i
    im = np.array(Image.open(im_path), dtype=np.uint8)
    # Start a figure to plot on and add the image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)

    # For each each expert annotation on that image
    for j in range(len(df_e.index)):

        # Make a polygon
        p_e = Polygon([(df_e.loc[j, "xmin"], df_e.loc[j, "ymin"]), (df_e.loc[j, "xmin"], df_e.loc[j, "ymax"]),
                     (df_e.loc[j, "xmax"], df_e.loc[j, "ymax"]), (df_e.loc[j, "xmax"], df_e.loc[j, "ymin"]),
                     (df_e.loc[j, "xmin"], df_e.loc[j, "ymin"])])
        # Save the polygon in the dataframe subset
        df_e.at[j, "polygon"] = p_e

        # Plot the polygon
        x_e,y_e = p_e.exterior.xy
        plt.plot(x_e,y_e, color='green')
        # Plot just the polygon center
        # plt.plot(df_e.loc[j, "x_center"], df_e.loc[j, "y_center"], color='green', marker='o')

    # Finalize plot settings and show
    ax.set_aspect('equal', adjustable='box')
    plt.xlim(0,512)
    plt.ylim(0,512)
    plt.gca().invert_yaxis()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

