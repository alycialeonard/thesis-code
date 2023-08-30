# Gets a csv of the classifications from a list of subjectids imported from CSV
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
from pathlib import Path

# Paths
subjectids_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\pttp_subjectids.csv"
classifications_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\DataExport_20200922\\panoptes\\All\\shape_extractor_all_prepped.csv"

# Read in files
classifications = pd.read_csv(classifications_path)
subjectids = pd.read_csv(subjectids_path)
subjectids_list = subjectids['subject_id'].to_list()

# Get rows only from filenames list
short_list = classifications[classifications['subject_id'].isin(subjectids_list)]

# Save short list as CSV
short_list.to_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\pttp_shape_extractor_raw.csv")