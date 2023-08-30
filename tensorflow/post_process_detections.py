# Post process detection CSVs into format where IOUs can be found.
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd

# Read in filenames and get as a list
filenames = pd.read_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R0\\expert_annotation_filenames.csv")
filenames_list = filenames['filenames'].to_list()

# Paths to data (csvs) and save
data_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R_test\\detections\\"
save_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\tensorflow_models\\2022-01-06_CrossVal_RawAnnots\\R_test\\detections\\processed\\"

# Have pandas always print all columns
pd.set_option('display.max_columns', None)

# Set IOU threshold for positive detection
iou_thresh = 0.5

# Make placeholder df for whole list.
final = pd.DataFrame(columns=['ymin', 'xmin', 'ymax', 'xmax', 'filename', 'scores', 'ymin_pixel', 'xmin_pixel', 'ymax_pixel', 'xmax_pixel'])

# For each file in the filenames list
for i in filenames_list:
    # Read the boxes file
    df = pd.read_csv(data_path + i + "_boxes2.csv", header=None, names=['ymin', 'xmin', 'ymax', 'xmax'])
    # Make column of just filename for each detection
    df['filename'] = i
    # Read the scores file and append to df
    scores = pd.read_csv(data_path + i + "_scores.csv", header=None).T
    df['scores'] = scores
    # Get corners in pixel values
    df['ymin_pixel'] = df['ymin'] * 512
    df['xmin_pixel'] = df['xmin'] * 512
    df['ymax_pixel'] = df['ymax'] * 512
    df['xmax_pixel'] = df['xmax'] * 512
    #print(df.head())
    #("Press Enter to continue...")
    # Save df to CSV
    df.to_csv(save_path + i + ".csv")
    # Concatenate it to the bigger CSV\
    final = pd.concat([final, df])

# Save final to csv
final.to_csv(save_path + "final_combined.csv")

