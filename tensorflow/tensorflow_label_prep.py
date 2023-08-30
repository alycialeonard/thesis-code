# Get list of file labels in test and train splits
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
import ast
import csv

pd.set_option('display.max_columns', 500)

# Helper functions
def get_color(details):
    details = ast.literal_eval(details)
    return max(details[0], key=details[0].get)


def get_shape(details):
    details = ast.literal_eval(details)
    return max(details[1], key=details[1].get)

# SPLIT OUT COLUMNS FOR COLOR AND SHAPE

# Data path
#data = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\tensorflow_train_labels.csv"

# Read data
#df = pd.read_csv(data)

#df["color"] = df["details"].apply(get_color)
#df["shape"] = df["details"].apply(get_shape)

#print(df.head())

#df.to_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\tensorflow_train_labels_classes.csv")


# SPLIT INTO TRAIN AND TEST SHEETS

# Data path
data = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\tensorflow_labels_minclustersize3.csv"
test_files = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\test_images_20210804.csv"
train_files = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\train_images_20210804.csv"

# Open data as dataframe and file lists as lists
df = pd.read_csv(data)

with open(train_files, newline='') as f:
    reader = csv.reader(f)
    train = list(reader)

# remove unnecessary brackets
train = [i[0] for i in train]

with open(test_files, newline='') as f:
    reader = csv.reader(f)
    test = list(reader)

# remove unnecessary brackets
test = [i[0] for i in test]

# Get dataframes with only the labels where the file is in those list
test_df = df[df['filename'].isin(test)]
train_df = df[df['filename'].isin(train)]

# Save those to csv
test_df.to_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\test_labels_20210804.csv")
train_df.to_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\train_labels_20210804.csv")

