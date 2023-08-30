# Count the number of annotations in an extracted classifications sheet
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
from pathlib import Path
import numpy as np
from ast import literal_eval

# Path to classifications
class_path = Path("C:/Users/alyci/Documents/DPhil/GeoDESA/Zooniverse/DataExports/DataExport_20200922/panoptes/All/shape_extractor_all_prepped.csv")

# Open classifications csv as dataframe
df = pd.read_csv(class_path)

# Replace NaN values in reduced with empty lists so the for loop later works (length of lists won't work on float)
df.replace(np.NaN, '[]', inplace=True)

# Instantiate counter to hold count of annotations
num_annotations = 0

# For each row in df
for i in range(len(df.index)):
    # Get the number of annotations in that row
    n_a = len(literal_eval(df.loc[i]["data.frame0.T1_tool0_x"]))
    # Add that to the tally
    num_annotations = num_annotations + n_a

# Print the result
print("This classification file includes " + str(num_annotations) + " shape annotations.")


