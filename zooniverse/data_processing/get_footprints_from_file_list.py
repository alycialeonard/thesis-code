# Gets a csv of the footprints from a list of files imported from a csv
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
from pathlib import Path

# Paths
filenames_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\split_9.csv"
full_list_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\all-annotations-prepped.csv"
# "D:\\fasterrcnn_20210909\\tensorflow_labels_minclustersize2.csv" "D:\\fasterrcnn_20210909\\imgs_9\\9.csv"

# Read in files
full_list = pd.read_csv(full_list_path)
filenames = pd.read_csv(filenames_path)
filenames_list = filenames['filename'].to_list()

# Get rows only from filenames list
short_list = full_list[full_list['filename'].isin(filenames_list)]

# Save short list as CSV
short_list.to_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\images_9_labels.csv")