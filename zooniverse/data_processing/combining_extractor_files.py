# This script combines multiple sets of extracted annotations produced with panoptes_aggregation
# It then combines the columns for each parameter for all frames
# E.g.: It will combine the x values for frame 0, frame 1, and frame 2 into one column.
# This is then saved to csv.
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
from pathlib import Path
import numpy as np
from ast import literal_eval

# Helper functions to combine things


def combine_x(row):
    return literal_eval(row["data.frame0.T1_tool0_x"]) + \
           literal_eval(row["data.frame1.T1_tool0_x"]) + \
           literal_eval(row["data.frame2.T1_tool0_x"])


def combine_y(row):
    return literal_eval(row["data.frame0.T1_tool0_y"]) + \
           literal_eval(row["data.frame1.T1_tool0_y"]) + \
           literal_eval(row["data.frame2.T1_tool0_y"])


def combine_width(row):
    return literal_eval(row["data.frame0.T1_tool0_width"]) + \
           literal_eval(row["data.frame1.T1_tool0_width"]) + \
           literal_eval(row["data.frame2.T1_tool0_width"])


def combine_height(row):
    return literal_eval(row["data.frame0.T1_tool0_height"]) + \
           literal_eval(row["data.frame1.T1_tool0_height"]) + \
           literal_eval(row["data.frame2.T1_tool0_height"])


def combine_angle(row):
    return literal_eval(row["data.frame0.T1_tool0_angle"]) + \
           literal_eval(row["data.frame1.T1_tool0_angle"]) + \
           literal_eval(row["data.frame2.T1_tool0_angle"])


def combine_details(row):
    return literal_eval(row["data.frame0.T1_tool0_details"]) + \
           literal_eval(row["data.frame1.T1_tool0_details"]) + \
           literal_eval(row["data.frame2.T1_tool0_details"])


# File paths
data_path = Path("C:/Users/alyci/Documents/DPhil/GeoDESA/Zooniverse/DataExports/DataExport_20200922/panoptes")
ken_extracted_path = data_path / "12362_Kenya/Combined/shape_extractor_rotateRectangle_kenya_12362.csv"
sle_extracted_path = data_path / "12791_SierraLeone/Combined/shape_extractor_rotateRectangle_sierraleone_12791.csv"
uga_extracted_path = data_path / "12809_Uganda/Combined/shape_extractor_rotateRectangle_uganda_12809.csv"

# Open these csvs in pandas
ken = pd.read_csv(ken_extracted_path)
sle = pd.read_csv(sle_extracted_path)
uga = pd.read_csv(uga_extracted_path)

# Append these dataframes to each other. No duplicate indices (i.e don' restart from 0, carry on from last df)
combined = ken.append([sle, uga], ignore_index=True)

# Replace NaN values with empty lists
combined.replace(np.NaN, '[]', inplace=True)

# Combine x, y, width, height, angle, details values for each frame into new column
combined['x'] = combined.apply(lambda row: combine_x(row), axis=1)
combined['y'] = combined.apply(lambda row: combine_y(row), axis=1)
combined['width'] = combined.apply(lambda row: combine_width(row), axis=1)
combined['height'] = combined.apply(lambda row: combine_height(row), axis=1)
combined['angle'] = combined.apply(lambda row: combine_angle(row), axis=1)
combined['details'] = combined.apply(lambda row: combine_details(row), axis=1)

pd.set_option('display.max_columns', None)
print(combined.head())

# Save to CSV
# combined.to_csv(data_path / "combined.csv")

#pd.set_option('display.max_columns', None)
#print(combined.head())









# Print lengths
#print("Length of KEN: ", len(ken.index), "\nLength of SLE: ", len(sle.index), "\nLength of UGA: ", len(uga.index))

# Check length of combined
#print("Length of combined: ", len(combined.index))

# s = pd.Series([[]], index=row.index)
# new = row["data.frame0.T1_tool0_x"].fillna(s) + \
#       row["data.frame1.T1_tool0_x"].fillna(s) + \
#       row["data.frame2.T1_tool0_x"].fillna(s)
# return new

# Turn NaN values into empty lists
#combined.loc[combined.isnull()] = combined.loc[combined.isnull()].apply(lambda x: [])

#combined['x'] = combined.apply(lambda row: combine_x(row), axis=1)
#combined['x'] = df.stack().groupby(level=0).sum()
#s = pd.Series([[]], index=combined.index)

#combined["x"] = literal_eval(combined["data.frame0.T1_tool0_x"]) + \
#                literal_eval(combined["data.frame1.T1_tool0_x"]) + \
#                literal_eval(combined["data.frame2.T1_tool0_x"])


