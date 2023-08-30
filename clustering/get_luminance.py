# Get the luminance of the three-band black marble image & save as raster
# Used in PSCC 2020 submission
# Alycia Leonard 2019-09-27

import sys
from osgeo.gdalconst import *
from osgeo import gdal
import numpy as np
import sys
from osgeo.gdalconst import *
from osgeo import gdal
import numpy as np

# Filepaths
black_marble = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\nasa_blackmarble\\colour\\BlackMarble_2016_C2_geo.tif'
luminance_fp = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\nasa_blackmarble\\colour\\BlackMarble_2016_C2_luminance.tif'

# Read in black marble data
night_gdata = gdal.Open(black_marble)
night_gt = night_gdata.GetGeoTransform()
night_data = night_gdata.ReadAsArray().astype(np.float)

# Get luminance matrix for night light data
night_data_rel_luminance = 0.2126*night_data[0] + 0.7152*night_data[1] + 0.0722*night_data[2]

# Save luminance as a raster

# register all of the GDAL drivers
gdal.AllRegister()

# Get number of rows and columns
night_gdata_rows = night_gdata.RasterYSize
night_gdata_cols = night_gdata.RasterXSize

# create the output image
driver = night_gdata.GetDriver()
# print driver
outDs = driver.Create(luminance_fp, night_gdata_cols, night_gdata_rows, 1, GDT_Int32)
if outDs is None:
    print('Could not create output file')
    sys.exit(1)

# Say where to write the output data, and what to write
outBand = outDs.GetRasterBand(1)
outData = night_data_rel_luminance

# write the data
outBand.WriteArray(outData, 0, 0)

# flush data to disk, set the NoData value and calculate stats
outBand.FlushCache()
outBand.SetNoDataValue(-99)

# georeference the image and set the projection
outDs.SetGeoTransform(night_gt)
outDs.SetProjection(night_gdata.GetProjection())

del outData
# Filepaths
black_marble = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\nasa_blackmarble\\colour\\BlackMarble_2016_C2_geo.tif'
luminance_fp = 'C:\\Users\\alyci\\Documents\\DPhil\\Transfer of Status\\Transfer_data\\nasa_blackmarble\\colour\\BlackMarble_2016_C2_luminance.tif'

# Read in black marble data
night_gdata = gdal.Open(black_marble)
night_gt = night_gdata.GetGeoTransform()
night_data = night_gdata.ReadAsArray().astype(np.float)

# Get luminance matrix for night light data
night_data_rel_luminance = 0.2126*night_data[0] + 0.7152*night_data[1] + 0.0722*night_data[2]

# Save luminance as a raster

# register all of the GDAL drivers
gdal.AllRegister()

# Get number of rows and columns
night_gdata_rows = night_gdata.RasterYSize
night_gdata_cols = night_gdata.RasterXSize

# create the output image
driver = night_gdata.GetDriver()
# print driver
outDs = driver.Create(luminance_fp, night_gdata_cols, night_gdata_rows, 1, GDT_Int32)
if outDs is None:
    print('Could not create output file')
    sys.exit(1)

# Say where to write the output data, and what to write
outBand = outDs.GetRasterBand(1)
outData = night_data_rel_luminance

# write the data
outBand.WriteArray(outData, 0, 0)

# flush data to disk, set the NoData value and calculate stats
outBand.FlushCache()
outBand.SetNoDataValue(-99)

# georeference the image and set the projection
outDs.SetGeoTransform(night_gt)
outDs.SetProjection(night_gdata.GetProjection())

del outData