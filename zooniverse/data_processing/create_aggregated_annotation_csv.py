# The purpose of this script is to move from the "reduced" clustered annotations CSV
# where each line is one subject_id containing multiple clusters
# to a CSV where each line is a single cluster (i.e. one aggregated annotation)
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
import numpy as np
from pathlib import Path
from ast import literal_eval

# File paths
data_path = Path("C:/Users/alyci/Documents/DPhil/GeoDESA/Zooniverse/DataExports/DataExport_20200922/panoptes")
reduced_path = data_path / "shape_reducer_hdbscan_all.csv"

# Open the "reduced" CSV in pandas
reduced = pd.read_csv(reduced_path)

# Replace NaN values in reduced with empty lists so the for loop later works (length of lists won't work on float)
reduced.replace(np.NaN, '[]', inplace=True)

# Define columns for new dataframe
columns = ["subject_id", "workflow", "x", "y", "width", "height", "angle", "details", "persistence", "count"]

# Instantiate empty dataframe to hold clusters
clusters = pd.DataFrame(columns=columns)

# For each row in the reduced dataframe
for i in range(len(reduced.index)):
    # For each cluster in that row:
    for j in range(len(literal_eval(reduced.loc[i]["data.frame0.T1_tool0_clusters_x"]))):
        # Create a dummy dataframe with the data of that cluster
        df = pd.DataFrame([[reduced.loc[i]["subject_id"],
                          reduced.loc[i]["workflow_id"],
                          literal_eval(reduced.loc[i]["data.frame0.T1_tool0_clusters_x"])[j],
                          literal_eval(reduced.loc[i]["data.frame0.T1_tool0_clusters_y"])[j],
                          literal_eval(reduced.loc[i]["data.frame0.T1_tool0_clusters_width"])[j],
                          literal_eval(reduced.loc[i]["data.frame0.T1_tool0_clusters_height"])[j],
                          literal_eval(reduced.loc[i]["data.frame0.T1_tool0_clusters_angle"])[j],
                          literal_eval(reduced.loc[i]["data.frame0.T1_tool0_clusters_details"])[j],
                          literal_eval(reduced.loc[i]["data.frame0.T1_tool0_clusters_persistance"])[j],
                          literal_eval(reduced.loc[i]["data.frame0.T1_tool0_clusters_count"])[j]]],
                          columns=columns)
        # Append the dummy dataframe to the clusters dataframe
        clusters = clusters.append(df, ignore_index=True)

pd.set_option('display.max_columns', None)
print(clusters.head())

# Save clusters to csv
clusters.to_csv(data_path / "shape_reducer_hdbscan_all_clusters.csv")







# print(reduced.loc[i])
# print(len(literal_eval(reduced.loc[i]["data.frame0.T1_tool0_clusters_x"])))
# print(reduced.loc[i]["data.frame0.T1_tool0_clusters_x"][j])

