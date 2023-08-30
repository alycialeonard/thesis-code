# Log of pre-processing steps for reference
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

from subprocess import call

# Set the parameters for retile
retile_script = 'C:/Users/alyci/Anaconda3/envs/gbdx/Lib/site-packages/GDAL-2.4.0-py3.6-win-amd64.egg-info/scripts/gdal_retile.py'
target_dir = 'C:/Users/alyci/Documents/DATA/gbdx/images/uganda/bidibidi/bidibidi_0_gdalpansharpened_tiles'
file = 'C:/Users/alyci/Documents/DATA/gbdx/images/uganda/bidibidi/bidibidi_0_gdalpansharpened.jpg'
size = '256'
# compression_format = 'COMPRESS=JPEG'

# Retile the image
# call(['python', retile_script, '-ps', size, size, '-co', compression_format, '-targetDir', target_dir, tif_file])
call(['python', retile_script, '-ps', size, size, '-targetDir', target_dir, '-of', 'JPEG', file])





# # Set the parameters for pansharpen
    # pansharpen_script = 'C:/Users/alyci/Anaconda3/envs/gbdx/Lib/site-packages/GDAL-2.4.0-py3.6-win-amd64.egg-info/scripts/gdal_pansharpen.py'
# pan_img = 'C:/Users/alyci/Documents/DATA/gbdx/images/uganda/bidibidi/bidibidi_0_pan_rendered_tif.tif'
# spectral_img = 'C:/Users/alyci/Documents/DATA/gbdx/images/uganda/bidibidi/bidibidi_0_RGB.tif'
# output_img = 'C:/Users/alyci/Documents/DATA/gbdx/images/uganda/bidibidi/bidibidi_0_gdalpansharpened.tif'

# Pansharpen the image
# call(['python', pansharpen_script, pan_img, spectral_img, output_img])




# Command line calls
# Run from inside GeoDESA-Env in terminal for gdal_retile.py to work
# Run mogrify from inside the folder with the tiles
# gdal_translate is to covert from float etc to byte (0-255) for web display

# Tile an image. Save georeferencing data to CSV
# gdal_retile.py -targetDir /Users/alycia/Downloads/test -csv output.csv -csvDelim , aoi_rgb.tif

# Convert all tifs in a folder to PNGs
# mogrify -format png *.tif

# Switch a signed 16-bit image to a 8-bit image
# gdal_translate -ot Byte -of GTiff -scale -32768 32767 0 255 SV1_20180515_PS.tif SV1_20180515_PS_byte.tif

# Switch the order of bands in an image
# gdal_translate -of GTiff -b 3 -b 2 -b 1 SV1_20180515_PS.tif SV1_20180515_PS_RGB.tif