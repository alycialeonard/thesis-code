# Pull imagery from Maxar GBDX
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.u

from gbdxtools.task import env
from gbdxtools import CatalogImage
import numpy as np

# GRAB THE IMAGERY WE WANT
# Machakos County, Kenya, WV02. Bands: Panchromatic and 8 MS (R, B, G, near-IR, red edge, coastal, yellow, near-IR2)

# Define Catalog ID (determined from GBDX online platform) and bounding box (in coordinates) for region we want to pull
catalog_id = '10300100645EB800'
bbox = [37.273736670613296, -1.5923564093044682, 37.29181258007885, -1.5726877317544576]

# Grab multispectral image of this region
image = CatalogImage(catalog_id, band_type="MS", bbox=bbox)

# Print multispectral image info
print("CatID: {} instantiated as a {} {}-band raster with {} rows and {} columns".format(catalog_id, image.dtype, *image.shape))
print("  with geographic bounds: ({})".format(bbox))
print("  in projection: {}".format(image.metadata["georef"]["spatialReferenceSystemCode"]))
print("  at {:0.2f} meter resolution".format(image.metadata["image"]["groundSampleDistanceMeters"]))
print("  and {:0.2f} degrees off nadir".format(image.metadata["image"]["offNadirAngle"]))
print("  taken on {}".format(image.metadata["image"]["acquisitionDate"]))
print("\n")

# Get the pansharpened image for this region
psharp = CatalogImage(catalog_id, pansharpen=True, bbox=bbox)

# Print pansharpened image info
print("CatID: {} instantiated as a {} {}-band raster with {} rows and {} columns".format(catalog_id, psharp.dtype, *psharp.shape))
print("  with geographic bounds: ({})".format(bbox))
print("  in projection: {}".format(psharp.metadata["georef"]["spatialReferenceSystemCode"]))
print("  at {:0.2f} meter resolution".format(psharp.metadata["image"]["groundSampleDistanceMeters"]))
print("  and {:0.2f} degrees off nadir".format(psharp.metadata["image"]["offNadirAngle"]))
print("  taken on {}".format(psharp.metadata["image"]["acquisitionDate"]))
print("\n")

# # Cover pansharpened image with 128x128 pixel windows and save these windows
# counter = 0
# for x in psharp.window_cover((128, 128)):
#     # Create unique path for saving this chip
#     save_path = 'images/window_%d.tif' % counter
#     # Save window
#     x.geotiff(path=save_path, proj="EPSG:4326")
#     # Increment counter
#     counter = counter + 1

