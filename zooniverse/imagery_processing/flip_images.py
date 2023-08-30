# Flip images that are in a given filenames list
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import pandas as pd
import PIL
from PIL import Image

# Path to images
image_path = "C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\visualisation\\"

# Read in filenames to visualise and get as a list
filenames = pd.read_csv("C:\\Users\\alyci\\Documents\\DPhil\\GeoDESA\\Zooniverse\\DataExports\\ExpertAnnotation_20210713\\finals_for_comparison\\filenames.csv")
filenames_list = filenames['filenames'].to_list()

# For each file in the filenames list
for i in filenames_list:
    #read the image
    im = Image.open(image_path + "annotations_" + i)
    #flip image
    out = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    out.save(image_path + 'flipped\\annnotation_flipped_' + i)

