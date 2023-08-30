# Join question counts to a data file on subject id
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
import numpy as np
from pathlib import Path
import json
from ast import literal_eval
import math

# File paths
data_path = Path("C:/Users/alyci/Documents/DPhil/GeoDESA/Zooniverse/DataExports")
questions_path = data_path / "question_counts_final_T2.csv"
annotations_path = data_path / "ExpertAnnotation_20210713/finals_for_comparison/contributor_clustered_footprints_minclustersize10_TEST.csv"

# Open the tiles, subjects, and footprints CSVs in pandas
questions = pd.read_csv(questions_path)
annotations = pd.read_csv(annotations_path)

# Join footprints and subjects data on subject_id
merged = pd.merge(left=annotations, right=questions, on='subject_id', how='left') # ['subject_id', 'workflow_id']

# Export merged dataframe
merged.to_csv(data_path / "ExpertAnnotation_20210713/finals_for_comparison/contributor_clustered_footprints_minclustersize10_TEST.csv")
