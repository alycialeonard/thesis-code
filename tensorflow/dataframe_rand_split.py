# Split dataframe into a number of random, even chunks
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)

# Data path
data = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\pttp-filenames-ps.csv"

# Set number of random samples to split into
n = 10

# Read data
df = pd.read_csv(data)

# Shuffle the data (random_state is selected so sample can be reproduced)
shuffled = df.sample(frac=1, random_state=1)
# Split into n equal samples
result = np.array_split(shuffled, n)

# Save each sample to csv
for i, sample in enumerate(result):
    sample.to_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\split_" + str(i) + ".csv")



