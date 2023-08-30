# Convert TIFF images to JPEGs
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

from PIL import Image
import os
import glob

directory = "itaa_samples/small"
crop_directory = "itaa_samples/small_crop"
resize_directory = "itaa_samples/small_crop_resize"
supp_directory = "bidibidi_supp"

# # Crop images in the director to half of their original size (centered)
# for filename in os.listdir(directory):
#     print(filename)
#     im = Image.open(directory + "/" + filename)
#     width, height = im.size
#     width_new, height_new = int(width/2), int(height/2)
#     left_corner, top_corner = int(width/4), int(height/4)
#     area = (left_corner, top_corner, left_corner + width_new, top_corner + height_new)
#     im_cropped = im.crop(area)
#     im_cropped.save(crop_directory + "/" + filename)

# # Resize cropped images to 32x32 resolution
# for filename in os.listdir(crop_directory):
#     print(filename)
#     im = Image.open(crop_directory + "/" + filename)
#     im_resized = im.resize((32, 32))  # defaults to nearest neighbour resampling filter))
#     im_resized.save(resize_directory + "/" + filename)

# Convert RGB .tif files to .jpg files. This works, but does not preserve image details.
# Source: https://github.com/kckaiwei/tifftojpeg/blob/master/main.py
def tif_to_jpeg(filepath):
    for name in glob.glob(filepath):
        im = Image.open(name)
        name = str(name).rstrip(".tif")
        im.save(name + '.jpg', 'JPEG')


tif_to_jpeg("C:/Users/alyci/PycharmProjects/GeoDESA/sample_processing/test_subjects/*.tif")
