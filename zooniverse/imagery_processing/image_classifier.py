# Script to classify images to prep for Zooniverse
# https://stackoverflow.com/questions/24662571/python-import-csv-to-list
# https://datascience.stackexchange.com/questions/14039/tool-to-label-images-for-classification
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import matplotlib.pyplot as plt
import csv
from PIL import Image

category = []
plt.ion()

input_csv = 'C:\\Users\\alyci\\Documents\\DATA\\earthi\\Echariria\\Echariria_20150715_PS_RGB_8bit_tiles_256\\Echariria_PNG_filenames.csv'
output_txt = 'C:\\Users\\alyci\\Documents\\DATA\\earthi\\Echariria\\Echariria_20150715_PS_RGB_8bit_tiles_256\\Echariria_labels.csv'
image_dir = 'C:\\Users\\alyci\\Documents\\DATA\\earthi\\Echariria\\Echariria_20150715_PS_RGB_8bit_tiles_256\\PNG\\'

# Open list of images from CSV
with open(input_csv, 'r') as f:
    reader = csv.reader(f)
    images = list(reader)

# Label image into category list
for i, image in enumerate(images):
    im = Image.open(image_dir + image[0])
    plt.imshow(im)
    plt.pause(0.05)
    category.append(input('category (y for possible houses, n for no houses): '))

# Write category to text file with each category on one line
with open(output_txt, 'w') as f:
    for item in category:
        f.write("%s\n" % item)



