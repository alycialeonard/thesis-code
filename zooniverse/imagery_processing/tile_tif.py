# Split TIFF images into tiles
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk


import os, gdal

in_path = 'C:/Users/Marco/Desktop/'
input_filename = 'dtm_5.tif'

out_path = 'C:/Users/Marco/Desktop/output_folder/'
output_filename = 'tile_'

tile_size_x = 50
tile_size_y = 70

ds = gdal.Open(in_path + input_filename)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

for i in range(0, xsize, tile_size_x):
    for j in range(0, ysize, tile_size_y):
        com_string = "gdal_translate -of GTIFF -srcwin " + str(i) + ", " + str(j) + ", " + str(
            tile_size_x) + ", " + str(tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(
            out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
        os.system(com_string)