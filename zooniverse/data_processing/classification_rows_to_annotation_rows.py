# Take Zooniverse extracted classifications, change the CSV from one row per classification to one row per annotation
# Same structure for clustered or for raw files basically, column names just change a bit
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
from pathlib import Path
import numpy as np
from ast import literal_eval

# Path to classifications
class_path = Path("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\pttp_shape_extractor_raw.csv")

# Path to save later
save_path = Path("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\pttp_annotations.csv")

# Open classifications csv as dataframe
df = pd.read_csv(class_path)

# Replace NaN values in reduced with empty lists so the for loop later works (length of lists won't work on float)
df.replace(np.NaN, '[]', inplace=True)

# Define columns for annotations dataframe
columns = ["subject_id", "classification_id", "user_name", "user_id", "x", "y", "width", "height", "angle", "details"]

# Instantiate empty dataframe to hold annotations
annotations = pd.DataFrame(columns=columns)

# For each row in df
for i in range(len(df.index)):
    # For each cluster in that row:

    for j in range(len(literal_eval(df.loc[i]["data.frame0.T1_tool0_x"]))):
        # Create a dummy dataframe with the data of that cluster
        dummy = pd.DataFrame([[df.loc[i]["subject_id"],
                               df.loc[i]["classification_id"],
                               df.loc[i]["user_name"],
                               df.loc[i]["user_id"],
                               literal_eval(df.loc[i]["data.frame0.T1_tool0_x"])[j],
                               literal_eval(df.loc[i]["data.frame0.T1_tool0_y"])[j],
                               literal_eval(df.loc[i]["data.frame0.T1_tool0_width"])[j],
                               literal_eval(df.loc[i]["data.frame0.T1_tool0_height"])[j],
                               literal_eval(df.loc[i]["data.frame0.T1_tool0_angle"])[j],
                               literal_eval(df.loc[i]["data.frame0.T1_tool0_details"])[j]]],
                             columns=columns)
        # Append the dummy dataframe to the clusters dataframe
        annotations = annotations.append(dummy, ignore_index=True)

# Save clusters to csv
annotations.to_csv(save_path)

