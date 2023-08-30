# This script creates PNG files (Pansharpened) cropped to each footprint in a CSV
# Not particularly useful, since it just gives you images cropped exactly to a roof, but hey!
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
import diagonal_crop
from PIL import Image
import math
import sys

# Declare paths to footprint csv, PNG images, and path to save footprint images
footprint_data_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\georeferenced_footprints_final.csv"
images_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\SubjectsPansharpened\\"
save_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\Footprints\\"

# Redirect console output to file
sys.stdout = open('file', 'w')

# Open footprints data
df = pd.read_csv(footprint_data_path)

# For each footprint, open containing image, save cropped image footprint image.
# Try and except for opening the image because not sure all image tiles downloaded locally.

print("Filenames for any images which would not open are printed:")

for index, row in df.iterrows():
    try:
        im = Image.open(images_path + row['filename'])
    except:
        print(row['filename'])
        continue
    angle = row['angle'] * math.pi/180
    base = (row['x1'], row['y1'])
    height = row['height']
    width = row['width']
    cropped_im = diagonal_crop.crop(im, base, angle, height, width)
    cropped_im.save(save_path + str(row["id"]) + ".png")

