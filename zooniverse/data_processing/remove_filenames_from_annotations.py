# remove annotations based on a list of filenames (CSV)
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd

# Read in filenames and get as a list
filenames = pd.read_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R_test\\expert_annotation_filenames.csv")
filenames_list = filenames['filename'].to_list()

# Load annotations as csv
df = pd.read_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R_test\\R_train_labels.csv")

# For each file in the filenames list
for i in filenames_list:
    # Drop those annotations and save in place
    df.drop(df.loc[df['filename']==i].index, inplace=True)

# Save the dataframe
df.to_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R_test\\R_train_labels_2.csv")
