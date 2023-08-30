# This script plots annotation clusters without needing to provide data as constants
# Simply change the subject_id and subject images directory
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.u

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
from ast import literal_eval

# PREP IF NEEDED: GET THE CLUSTERS WITH SUBJECT INFO

# def get_filename(meta):
#     return literal_eval(meta)["Pansharpened"]
#
#
# # Paths
# clusters_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\DataExport_20200922\\panoptes\\All\\shape_reducer_hdbscan_all_clusters.csv"
# subjects_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\DataExport_20201115\\power-to-the-people-subjects-live.csv"
#
# # Open the clusters and subjects csvs in pandas
# clusters = pd.read_csv(clusters_path)
# subjects = pd.read_csv(subjects_path)
#
# # Join clusters and subjects data on subject_id
# clusters = pd.merge(left=clusters, right=subjects, on='subject_id', how='left') # ['subject_id', 'workflow_id']
# # Extract filename into separate column
# clusters['filename'] = clusters["metadata"].apply(get_filename)
#
# # Print head of clusters
# pd.set_option('display.max_columns', None)
# print(clusters.head())
#
# # Save to csv
# clusters.to_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\DataExport_20200922\\panoptes\\All\\shape_reducer_hdbscan_all_clusters_with_subject_data.csv")

# Read csv of clusters with subject info now that it exists
clusters = pd.read_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\DataExport_20200922\\panoptes\\All\\shape_reducer_hdbscan_all_clusters_with_subject_data.csv")

# Define subject ID to visualise
subject_id = 40169627

# Define where to find images for that subject set
subject_imgs_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\SubjectSets\\Alikalia_20180403_SubjectSet\\Alikalia_20180403_SubjectSet"

# Grab the image for the defined subject ID
clusters_for_vis = clusters.loc[clusters['subject_id'] == subject_id].copy()
print(clusters_for_vis.head())
image_name = clusters_for_vis['filename']
image_path = subject_imgs_path + "\\" + image_name.values[0]

# Load the image for the defined subject_id
im = np.array(Image.open(image_path), dtype=np.uint8)

# Create figure and axes
fig, ax = plt.subplots(1)
plt.axis('off')

# Display the image
ax.imshow(im)
plt.show(block=False)

fig, ax = plt.subplots(1)
plt.axis('off')
ax.imshow(im)

# Print all the clusters on the image
for index, row in clusters_for_vis.iterrows():
    # Create a Rectangle patch corresponding to the cluster
    rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], row['angle'], linewidth=2, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

plt.show()



