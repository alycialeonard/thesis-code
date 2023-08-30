# Get a csv of the sizes of all image tiles in a directory
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import glob
from PIL import Image
import pandas as pd

# Define columns for new dataframe
columns = ["filename", "w", "h"]

# Instantiate empty dataframe to hold image size data
img_df = pd.DataFrame(columns=columns)

# Grab all the images in the folder
images = glob.glob("Images/*.png")
for image in images:
    with open(image, 'rb') as file:
        img = Image.open(file)
        w, h = img.size
        fn = img.filename
        # Dummy dataframe to hold data about this image
        df = pd.DataFrame([[fn, w, h]], columns=columns)
        # Append the dummy dataframe to the img_df dataframe
        img_df = img_df.append(df, ignore_index=True)

# Save image size dataframe to csv
img_df.to_csv("x.csv")


