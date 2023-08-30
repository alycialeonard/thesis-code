# Get a random split of a dataframe (e.g. for test/train split).
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk


import pandas as pd

pd.set_option('display.max_columns', 500)

# Data path
data = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\tensorflow_labels_minclustersize3_filenames.csv"

# Set fraction to split out randomly
f = 0.2  # 20% going to test set

# Read data
df = pd.read_csv(data)

# Get random sample for expert annotation AND tensorflow test (random_state is selected so sample can be reproduced)
frac_1 = df.sample(frac=f, random_state=1)
# Save to CSV
frac_1.to_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\test_images_20210804.csv")

# Get the rest of the images not in the sample which will be the training set
frac_2 = pd.concat([df, frac_1]).drop_duplicates(keep=False)
# Save to CSV
frac_2.to_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\train_images_20210804.csv")


