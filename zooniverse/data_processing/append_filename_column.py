# Append filename column from subjects to annotations list
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
from pathlib import Path
from ast import literal_eval

def get_filename(meta):
    return literal_eval(meta)["Pansharpened"]

# Paths
annotations_path = "C:\\Users\\alyci\\OneDrive - Nexus365\\Data\\Zooniverse\\DataExports\\DataForRepository\\pttp-annotations-georeferenced.csv"
subjects_path = "C:\\Users\\alyci\\OneDrive - Nexus365\\Data\\Zooniverse\\DataExports\\DataForRepository\\pttp-subjects.csv"

# Read in files
annotations = pd.read_csv(annotations_path)
subjects = pd.read_csv(subjects_path)

# Extract filename into separate column in subjects
subjects['filename'] = subjects["metadata"].apply(get_filename)

# Join filename to annotations on subject_id
#annotations = annotations.join(subjects, on='subject_id', how='left', lsuffix='_left', rsuffix='_right')
merged = pd.merge(left=annotations, right=subjects, on='subject_id', how='left')

# Save to file
merged.to_csv("C:\\Users\\alyci\\OneDrive - Nexus365\\Data\\Zooniverse\\DataExports\\DataForRepository\\pttp-annotations-georeferenced-filenames.csv")