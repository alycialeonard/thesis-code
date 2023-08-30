# Gets a csv of subjectids a list of files imported from a csv
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
from pathlib import Path
from ast import literal_eval


def get_filename(meta):
    return literal_eval(meta)["Pansharpened"]


# Paths
filenames_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\filenames.csv"
subjects_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\DataExport_20201115\\power-to-the-people-subjects-live.csv"

# Read in files
subjects = pd.read_csv(subjects_path)
filenames = pd.read_csv(filenames_path)
filenames_list = filenames['filenames'].to_list()

# Extract filename into separate column
subjects['filename'] = subjects["metadata"].apply(get_filename)

# Get rows only from filenames list
short_list = subjects[subjects['filename'].isin(filenames_list)]

# Save short list as CSV
short_list.to_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\pttp_subjects.csv")