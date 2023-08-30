# Resample a tiff file
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import rasterio
from rasterio.enums import Resampling

input_tiff = "C:\\Users\\alyci\\Documents\\DATA\\earthi\\Alikalia\\ATCOR_oSV1-02_20180403_L2A0000837987_11U2190013004_01-MUX_RGB_8bit_crop.tif"
resample_factor_width = 4.0011942675159
resample_factor_height = 4
dest_tiff = "C:\\Users\\alyci\\Documents\\DATA\\earthi\\Alikalia\\ATCOR_oSV1-02_20180403_L2A0000837987_11U2190013004_01-MUX_RGB_8bit_crop_upsampled.tif"

dat = rasterio.open(input_tiff)

with dat as dataset:
    # resample data to target shape
    data = dataset.read(
        out_shape=(dataset.count, int(dataset.width * resample_factor_width), int(dataset.height * resample_factor_height)),
        resampling=Resampling.bilinear
    )
    # scale image transform
    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-2]), (dataset.height / data.shape[-1]))

# Write resampled tiff to file
with rasterio.open(
    dest_tiff,
    'w',
    driver='GTiff',
    count=data.shape[0],
    width=data.shape[1],
    height=data.shape[2],
    dtype=data.dtype,
    crs=dat.crs,
    transform=transform,
) as dst:
    dst.write(data)



