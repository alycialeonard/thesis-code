# Append georeferencing data to subjects (or annotations)
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
from pathlib import Path
from ast import literal_eval


def get_filename(meta):
    return literal_eval(meta)["Pansharpened"]


# File paths
data_path = Path("C:/Users/alyci/Documents/DPhil/GeoDESA")

tiles_path = data_path / "PS_Tiles_Georeference.csv"
subjects_path = data_path / "all_annotations.csv"

# Open the tiles, subjects, and footprints CSVs in pandas
georef = pd.read_csv(tiles_path)
subjects = pd.read_csv(subjects_path)

# Extract filename into separate column
#subjects['filename'] = subjects["metadata"].apply(get_filename)
# Join geo-referencing data for the subject on the filename
merged = pd.merge(left=subjects, right=georef, on='filename', how='left')

# Export
merged.to_csv(data_path / "all-annotations-2.csv")