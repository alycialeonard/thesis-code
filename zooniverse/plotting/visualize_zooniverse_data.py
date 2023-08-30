# Visualise raw annotations, clusters, expert annotations if present for files in a list
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

from shapely.geometry import Polygon
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
from ast import literal_eval
from math import cos, sin, pi

# Path to images
image_path = "C:\\Users\\alyci\\OneDrive - Nexus365\\Data\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\expert_annotation_subjects\\"

# Read in filenames to visualise and get as a list
filenames = pd.read_csv("C:\\Users\\alyci\\OneDrive - Nexus365\\Data\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\filenames.csv")
filenames_list = filenames['filenames'].to_list()

# Load expert and contributor footprints + all annotations as csv
expert = pd.read_csv("C:\\Users\\alyci\\OneDrive - Nexus365\\Data\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\all_extracted_expert_annotation_footprints_corners_TEST.csv")
clusters = pd.read_csv("C:\\Users\\alyci\\OneDrive - Nexus365\\Data\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\contributor_clustered_footprints_minclustersize2_TEST.csv")
classifications = pd.read_csv("C:\\Users\\alyci\OneDrive - Nexus365\\Data\\Zooniverse\\DataExports\\DataExport_20200922\\panoptes\\All\\shape_extractor_all_prepped.csv")

# Have pandas always print all columns
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = 'raise'

# For each file in the filenames list
for i in filenames_list:

    # Get subset of annotations that correspond to that filename
    df_e = expert.loc[expert['filename'] == i]
    df_c = clusters.loc[clusters['filename'] == i]
    # Reset index of subset to go from 0 to length
    df_e = df_e.reset_index(drop=True)
    df_c = df_c.reset_index(drop=True)
    # Add empty column to hold polygons
    df_e['polygon'] = np.nan
    df_c['polygon'] = np.nan

    # Get power to the people subject_id for that filename
    subject = df_c.loc[0, "subject_id"]
    # Get all classifications for that subject id as a dataframe
    df_class = classifications.loc[classifications['subject_id'] == subject]
    # Replace NaN values in reduced with empty lists so the for loop later works (length of lists won't work on float)
    df_class = df_class.replace(np.NaN, '[]', inplace=False)

    # Define columns for annotations dataframe
    columns = ["subject_id", "classification_id", "user_name", "user_id", "x", "y", "width", "height", "angle", "details"]
    # Instantiate empty dataframe to hold annotations
    df_a = pd.DataFrame(columns=columns)
    # For each row in the classifications for that subject_id
    for j in range(len(df_class.index)):
        # For each cluster in that row:
        for k in range(len(literal_eval(df_class.iloc[j]["data.frame0.T1_tool0_x"]))):
            # Create a dummy dataframe with the data of that cluster
            dummy = pd.DataFrame([[df_class.iloc[j]["subject_id"],
                                   df_class.iloc[j]["classification_id"],
                                   df_class.iloc[j]["user_name"],
                                   df_class.iloc[j]["user_id"],
                                   literal_eval(df_class.iloc[j]["data.frame0.T1_tool0_x"])[k],
                                   literal_eval(df_class.iloc[j]["data.frame0.T1_tool0_y"])[k],
                                   literal_eval(df_class.iloc[j]["data.frame0.T1_tool0_width"])[k],
                                   literal_eval(df_class.iloc[j]["data.frame0.T1_tool0_height"])[k],
                                   literal_eval(df_class.iloc[j]["data.frame0.T1_tool0_angle"])[k],
                                   literal_eval(df_class.iloc[j]["data.frame0.T1_tool0_details"])[k]]],
                                 columns=columns)
            # Append the dummy dataframe to the clusters dataframe
            df_a = df_a.append(dummy, ignore_index=True)

    # And now we get the corners of those annotations
    df_a['x1_unrotated'] = df_a['x']
    df_a['y1_unrotated'] = df_a['y']
    df_a['x2_unrotated'] = df_a.apply(lambda row: row.x + row.width, axis=1)
    df_a['y2_unrotated'] = df_a['y']
    df_a['x3_unrotated'] = df_a.apply(lambda row: row.x + row.width, axis=1)
    df_a['y3_unrotated'] = df_a.apply(lambda row: row.y + row.height, axis=1)
    df_a['x4_unrotated'] = df_a['x']
    df_a['y4_unrotated'] = df_a.apply(lambda row: row.y + row.height, axis=1)
    df_a['x_center'] = df_a.apply(lambda row: row.x + (row.width / 2), axis=1)
    df_a['y_center'] = df_a.apply(lambda row: row.y + (row.height / 2), axis=1)
    # Then, rotate each point
    df_a['x1'] = df_a.apply(lambda row: ((row.x1_unrotated - row.x_center)*cos(-row.angle * pi/180) + (row.y1_unrotated - row.y_center)*sin(-row.angle * pi/180)) + row.x_center, axis=1)
    df_a['y1'] = df_a.apply(lambda row: (-(row.x1_unrotated - row.x_center)*sin(-row.angle * pi/180) + (row.y1_unrotated - row.y_center)*cos(-row.angle * pi/180)) + row.y_center, axis=1)
    df_a['x2'] = df_a.apply(lambda row: ((row.x2_unrotated - row.x_center)*cos(-row.angle * pi/180) + (row.y2_unrotated - row.y_center)*sin(-row.angle * pi/180)) + row.x_center, axis=1)
    df_a['y2'] = df_a.apply(lambda row: (-(row.x2_unrotated - row.x_center)*sin(-row.angle * pi/180) + (row.y2_unrotated - row.y_center)*cos(-row.angle * pi/180)) + row.y_center, axis=1)
    df_a['x3'] = df_a.apply(lambda row: ((row.x3_unrotated - row.x_center)*cos(-row.angle * pi/180) + (row.y3_unrotated - row.y_center)*sin(-row.angle * pi/180)) + row.x_center, axis=1)
    df_a['y3'] = df_a.apply(lambda row: (-(row.x3_unrotated - row.x_center)*sin(-row.angle * pi/180) + (row.y3_unrotated - row.y_center)*cos(-row.angle * pi/180)) + row.y_center, axis=1)
    df_a['x4'] = df_a.apply(lambda row: ((row.x4_unrotated - row.x_center)*cos(-row.angle * pi/180) + (row.y4_unrotated - row.y_center)*sin(-row.angle * pi/180)) + row.x_center, axis=1)
    df_a['y4'] = df_a.apply(lambda row: (-(row.x4_unrotated - row.x_center)*sin(-row.angle * pi/180) + (row.y4_unrotated - row.y_center)*cos(-row.angle * pi/180)) + row.y_center, axis=1)
    # Add empty column to hold polygons
    df_a['polygon'] = np.nan

    # Get path to that image and load it as a numpy array
    im_path = image_path + i
    im = np.array(Image.open(im_path), dtype=np.uint8)
    # What size does the figure need to be in inches to fit the image?
    dpi = mpl.rcParams['figure.dpi']
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape
    figsize = width / float(dpi), height / float(dpi)
    # Start a figure to plot on and add the image
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.imshow(im)

    # For each raw annotation on that image
    for j in range(len(df_a.index)):
        # Make a polygon
        p_a = Polygon([(df_a.loc[j, "x1"], df_a.loc[j, "y1"]), (df_a.loc[j, "x2"], df_a.loc[j, "y2"]),
                        (df_a.loc[j, "x3"], df_a.loc[j, "y3"]), (df_a.loc[j, "x4"], df_a.loc[j, "y4"]),
                        (df_a.loc[j, "x1"], df_a.loc[j, "y1"])])
        # Save the polygon in the dataframe subset
        df_a.at[j, "polygon"] = p_a
        # Plot the polygon
        x_a, y_a = p_a.exterior.xy
        plt.plot(x_a, y_a, color='yellow', alpha=0.5)

    # For each each expert annotation on that image
    for j in range(len(df_e.index)):
        # Make a polygon
        p_e = Polygon([(df_e.loc[j, "x1"], df_e.loc[j, "y1"]), (df_e.loc[j, "x2"], df_e.loc[j, "y2"]),
                     (df_e.loc[j, "x3"], df_e.loc[j, "y3"]), (df_e.loc[j, "x4"], df_e.loc[j, "y4"]),
                     (df_e.loc[j, "x1"], df_e.loc[j, "y1"])])
        # Save the polygon in the dataframe subset
        df_e.at[j, "polygon"] = p_e
        # Plot the polygon
        # x_e,y_e = p_e.exterior.xy
        # plt.plot(x_e,y_e, color='green', alpha=0.8)

    # For each contributor footprint on that image
    for j in range(len(df_c.index)):
        # Make a polygon
        p_c = Polygon([(df_c.loc[j, "x1"], df_c.loc[j, "y1"]), (df_c.loc[j, "x2"], df_c.loc[j, "y2"]),
                     (df_c.loc[j, "x3"], df_c.loc[j, "y3"]), (df_c.loc[j, "x4"], df_c.loc[j, "y4"]),
                     (df_c.loc[j, "x1"], df_c.loc[j, "y1"])])
        # Save the polygon in the dataframe subset
        df_c.at[j, "polygon"] = p_c
        # Plot the polygon
        # x_c,y_c = p_c.exterior.xy
        # plt.plot(x_c,y_c, color='red', alpha=0.8)

    # Finalize plot settings and show
    ax.set_aspect('equal', adjustable='box')
    plt.xlim(0,512)
    plt.ylim(0,512)
    #plt.gca().invert_yaxis()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig('C:\\Users\\alyci\OneDrive - Nexus365\\Data\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\visualisation_2\\annotations_' + i, bbox_inches='tight')
    plt.close(fig)
    # plt.show()

# Flip all images to their original orientation.

# Path to images
image_path = "C:\\Users\\alyci\OneDrive - Nexus365\\Data\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\visualisation_2\\"

# Read in filenames to visualise and get as a list
filenames = pd.read_csv("C:\\Users\\alyci\OneDrive - Nexus365\\Data\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\filenames.csv")
filenames_list = filenames['filenames'].to_list()

# For each file in the filenames list
for i in filenames_list:
    # read the image
    im = Image.open(image_path + "annotations_" + i)
    # flip image and save
    out = im.transpose(Image.FLIP_TOP_BOTTOM)
    out.save(image_path + 'flipped\\annnotation_flipped_' + i)