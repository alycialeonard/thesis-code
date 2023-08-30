# Pull data from Maxar GBDX
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

from gbdxtools import CatalogImage
from shapely.geometry import Point
import csv

# Define Catalog ID and bounding box (in decimal coordinates) for imagery we want to pull
# WV02 Satellite Spec: https://www.satimagingcorp.com/satellite-sensors/worldview-2/
# WV03 Satellite Spec: https://www.satimagingcorp.com/satellite-sensors/worldview-3/
# WV04 Satellite Spec: https://www.satimagingcorp.com/satellite-sensors/geoeye-2/

# Details listed below for each image are also in image_info.csv

"""
# Bidibidi 0 - near Yumbe Uganda, containing coords provided by MM with least possible cloud cover
# Image spec: 0% Clouds, Sun Azimuth 135.24756, Sun Elevation 60.905056, Off Nadir Angle 6.2023797
# Catalog ID 1040010029149500, WV03, taken 1 Feb 2017 at 8:44:09 AM
"""
bidibidi_0_id = '1040010029149500'
bidibidi_0_bbox = [31.369, 3.473, 31.372, 3.475]

"""
# Bidibidi 1 - near Bidi bidi 0, region with many visible homes. No WV03 for this region
# Image spec: 1% Clouds, Sun Azimuth 111.60752, Sun Elevation 59.41518, Off Nadir Angle 40.75454
# Catalog ID 1030010065B80F00, WV02, taken 28 Feb 2017 at 8:13:06 AM
"""
bidibidi_1_id = '1030010065B80F00'
bidibidi_1_bbox = [31.3578987119945, 3.4506810991162715, 31.36697959958838, 3.460444110126974]

"""
# Bidibidi 2 - next to Bidi bidi 1, right next to Bidibidi 1, less homes but WV03 available.
# Image spec: 1% Clouds, Sun Azimuth 74.64639, Sun Elevation 70.77652, Off Nadir Angle 16.986647
# Catalog ID 104001002C746300, WV03, taken 10 April 2017 at 9:41:12 AM
"""
bidibidi_2_id = '104001002C746300'
bidibidi_2_bbox = [31.348925113416048, 3.4506895039643886, 31.357817172611252, 3.4604868318804396]

"""
# Bidibidi 3 - near Bidi bidi 0, shows some green space
# Image spec: 0% Clouds, Sun Azimuth 135.24756, Sun Elevation 60.905056, Off Nadir Angle 6.2023797
# Catalog ID 1040010029149500, WV03, taken 1 Feb 2017 at 8:44:09 AM
"""
bidibidi_3_id = '1040010029149500'
bidibidi_3_bbox = [31.384875297644612, 3.4520648550750987, 31.39860820796458, 3.4656884121522533]

"""
# Bidibidi 4 - near Bidi bidi 3, north and west, shows more homes. Same catalog id
# Image spec: 0% Clouds, Sun Azimuth 135.24756, Sun Elevation 60.905056, Off Nadir Angle 6.2023797
# Catalog ID 1040010029149500, WV03, taken 1 Feb 2017 at 8:44:09 AM
"""
bidibidi_4_id = '1040010029149500'
bidibidi_4_bbox = [31.37285041782889, 3.465874243183528, 31.384626388680775, 3.4773014046805995]

"""
# Bidibidi 5 - Another one. Same catalog id!
# Image spec: 0% Clouds, Sun Azimuth 135.24756, Sun Elevation 60.905056, Off Nadir Angle 6.2023797
# Catalog ID 1040010029149500, WV03, taken 1 Feb 2017 at 8:44:09 AM
"""
bidibidi_5_id = '1040010029149500'
bidibidi_5_bbox = [31.385630607474017, 3.4672505739166186, 31.39932918522391, 3.479329716379249]

"""
# Bidibidi 6 - Another bit.     
# Image spec: 0% Clouds, Sun Azimuth 135.24756, Sun Elevation 60.905056, Off Nadir Angle 6.2023797
# Catalog ID 1040010029149500, WV03, taken 1 Feb 2017 at 8:44:09 AM
"""
bidibidi_6_id = '1040010029149500'
bidibidi_6_bbox = [31.357563972014766, 3.460932868995396, 31.369030952191682, 3.471502199397291]

"""
# Yumbe 0 - denser cityish region near other bidi bidi samples
# Image spec: 1% Clouds, Sun Azimuth 74.64639, Sun Elevation 70.77652, Off Nadir Angle 16.986647
# Catalog ID 104001002C746300, WV03, taken 10 April 2017 at 9:41:12 AM
"""
yumbe_0_id = '104001002C746300'
yumbe_0_bbox = [31.24167108529833, 3.462380123295179, 31.25250291817793, 3.4740132224712794]

"""
# Makueni 0 - region near Maximus' coordinates with DigitalGlobe available
# Image spec: 0% Clouds, Sun Azimuth 118.31976, Sun Elevation 62.977818, Off Nadir Angle 22.397478
# Catalog ID 1030010065240F00, WV02, taken 12 Feb 2017 at 8:06:55 AM
"""
makueni_0_id = '1030010065240F00'
makueni_0_bbox = [37.568049430847175, -1.7904532654063985, 37.57771396623867, -1.7791137355330098]

"""
# Makueni 1 - another region near Maximus' coordinates with DigitalGlobe available (same catalog image)
# Image spec: 0% Clouds, Sun Azimuth 118.31976, Sun Elevation 62.977818, Off Nadir Angle 22.397478
# Catalog ID 1030010065240F00, WV02, taken 12 Feb 2017 at 8:06:55 AM
"""
makueni_1_id = '1030010065240F00'
makueni_1_bbox = [37.5557756419039, -1.7625991472075988, 37.56645297968135, -1.7510704509155701]

"""
# Makueni 2 - another region near Maximus' coordinates with DigitalGlobe available (same catalog image)
# Image spec: 0% Clouds, Sun Azimuth 118.31976, Sun Elevation 62.977818, Off Nadir Angle 22.397478
# Catalog ID 1030010065240F00, WV02, taken 12 Feb 2017 at 8:06:55 AM
"""
makueni_2_id = '1030010065240F00'
makueni_2_bbox = [37.54745864789584, -1.9184750304582594, 37.567302702809684, -1.8967596105214073]

"""
# Makueni 3 - another region near Maximus' coordinates with DigitalGlobe available (same catalog image)
# Image spec: 0% Clouds, Sun Azimuth 118.31976, Sun Elevation 62.977818, Off Nadir Angle 22.397478
# Catalog ID 1030010065240F00, WV02, taken 12 Feb 2017 at 8:06:55 AM
"""
makueni_3_id = '1030010065240F00'
makueni_3_bbox = [37.553861617616356, -1.8076842670862951, 37.568727493580816, -1.7914997954845764]

"""
# Makueni 4 - a more clustered community in Makueni region
# Image spec: 1% Clouds, Sun Azimuth 117.90515, Sun Elevation 62.572197, Off Nadir Angle 6.357525
# Catalog ID 10300100645EB800, WV02, taken 12 Feb 2017 at 8:06:06 AM
"""
makueni_4_id = '10300100645EB800'
makueni_4_bbox = [37.25199508591687, -1.857677080349614, 37.267547606898006, -1.8453758643448117]

"""
# Makueni 5 - another region with more "green" areas
# Image spec: 0% Clouds, Sun Azimuth 118.31976, Sun Elevation 62.977818, Off Nadir Angle 22.397478
# Catalog ID 1030010065240F00, WV02, taken 12 Feb 2017 at 8:06:55 AM
"""
makueni_5_id = '1030010065240F00'
makueni_5_bbox = [37.465301514457686, -1.703401474953499, 37.48336029144412, -1.683882795606190]

"""
# Makueni 6 - another region in the other "end" of the county
# Image spec: 0% Clouds, Sun Azimuth 136.903045654297, Sun Elevation 75.0801544189453, Off Nadir Angle 27.5306072235107
# Catalog ID 103001001CB1E500, WV02, taken 28 Oct 2012 at 8:29:44 AM
"""
makueni_6_id = '103001001CB1E500'
makueni_6_bbox = [38.069068908544075, -2.604583262358061, 38.08475875841396, -2.5908458370817766]

"""
# Makueni 7 - another region in the other "end" of the county
# Image spec: 4% Clouds, Sun Azimuth 135.95, Sun Elevation 60.819, Off Nadir Angle 19.927134
# Catalog ID 1030010077CD2500, WV02, taken 9 Jan 2018 at 8:09:53 AM
"""
makueni_7_id = '1030010077CD2500'
makueni_7_bbox = [37.833815575759225, -2.4145706091376766, 37.84315395525482, -2.404987380532951]

"""
# Makueni 8 - another chunk
# Image spec: 4% Clouds, Sun Azimuth 135.95, Sun Elevation 60.819, Off Nadir Angle 19.927134
# Catalog ID 1030010077CD2500, WV02, taken 9 Jan 2018 at 8:09:53 AM
"""
makueni_8_id = '1030010077CD2500'
makueni_8_bbox = [37.795784003392335, -2.0473697266365485, 37.8372573904926, -1.9956960541272084]

"""
# Makueni 9 - another chunk
# Image spec: 4% Clouds, Sun Azimuth 135.95, Sun Elevation 60.819, Off Nadir Angle 19.927134
# Catalog ID 1030010077CD2500, WV02, taken 9 Jan 2018 at 8:09:53 AM
"""
makueni_9_id = '1030010077CD2500'
makueni_9_bbox = [37.79310607700609, -1.8687786702366986, 37.809, -1.85]

"""
# Makueni 10 - another region 
# Image spec: 0% Clouds, Sun Azimuth 118.31976, Sun Elevation 62.977818, Off Nadir Angle 22.397478
# Catalog ID 1030010065240F00, WV02, taken 12 Feb 2017 at 8:06:55 AM
"""
makueni_10_id = '1030010065240F00'
makueni_10_bbox = [37.544523239921546, -1.99, 37.56937980651856, -1.9708699568363077]

"""
# Makueni 11 - another region 
# Image spec: 0% Clouds, Sun Azimuth 118.31976, Sun Elevation 62.977818, Off Nadir Angle 22.397478
# Catalog ID 1030010065240F00, WV02, taken 12 Feb 2017 at 8:06:55 AM
"""
makueni_11_id = '1030010065240F00'
makueni_11_bbox = [37.53739929408767, -1.855553432361662, 37.56101990060416, -1.8297138765203227]

"""
# Freetown 0 - Central freetown sample
# Image spec: 0% Clouds, Sun Azimuth 147.30663, Sun Elevation 58.131786, Off Nadir Angle 23.751455
# Catalog ID 1040010046199500, WV03, taken 27 Jan 2019 at 11:55:37 AM
"""
freetown_0_id = '1040010046199500'
freetown_0_bbox = [-13.256970404563617, 8.46613738862592, -13.239460943950691, 8.482935129296147]

"""
# Freetown 1 - South and west of Freetown 0, same catalog id
# Image spec: 0% Clouds, Sun Azimuth 147.30663, Sun Elevation 58.131786, Off Nadir Angle 23.751455
# Catalog ID 1040010046199500, WV03, taken 27 Jan 2019 at 11:55:37 AM
"""
freetown_1_id = '1040010046199500'
freetown_1_bbox = [-13.272943498013776, 8.437653549016005, -13.255983354101774, 8.454724575135083]

"""
# Freetown 2 - East of Freetown 0
# Image spec: 0% Clouds, Sun Azimuth 147.4825, Sun Elevation 58.164, Off Nadir Angle 21.444992
# Catalog ID 1040010047330800, WV03, taken 27 Jan 2019 at 11:56:02 AM
"""
freetown_2_id = '1040010047330800'
freetown_2_bbox = [-13.195730209358773, 8.451923433524314, -13.177568435930883, 8.471612126174087]

"""
# Freetown 3 - South-east of central (same catalog_id)
# Image spec: 0% Clouds, Sun Azimuth 147.4825, Sun Elevation 58.164, Off Nadir Angle 21.444992
# Catalog ID 1040010047330800, WV03, taken 27 Jan 2019 at 11:56:02 AM
"""
freetown_3_id = '1040010047330800'
freetown_3_bbox = [-13.24, 8.434965912537052, -13.225238790619189, 8.45]

"""
# Freetown 4 - Another eastern piece (same catalog_id)
# Image spec: 0% Clouds, Sun Azimuth 147.4825, Sun Elevation 58.164, Off Nadir Angle 21.444992
# Catalog ID 1040010047330800, WV03, taken 27 Jan 2019 at 11:56:02 AM
"""
freetown_4_id = '1040010047330800'
freetown_4_bbox = [-13.211591718354615, 8.460625030957164, -13.197446820631741, 8.48000725145232]

"""
# Freetown 5 - Another western piece (same catalog_id)
# Image spec: 0% Clouds, Sun Azimuth 147.4825, Sun Elevation 58.164, Off Nadir Angle 21.444992
# Catalog ID 1040010047330800, WV03, taken 27 Jan 2019 at 11:56:02 AM
"""
freetown_5_id = '1040010047330800'
freetown_5_bbox = [-13.288590432784988, 8.423291253237082, -13.279149057052566, 8.434173673079703]

"""
# Freetown 6 - Another central piece (same catalog_id)
# Image spec: 0% Clouds, Sun Azimuth 147.4825, Sun Elevation 58.164, Off Nadir Angle 21.444992
# Catalog ID 1040010047330800, WV03, taken 27 Jan 2019 at 11:56:02 AM
"""
freetown_6_id = '1040010047330800'
freetown_6_bbox = [-13.276514052558925, 8.466029962420814, -13.259691236307848, 8.484731857441576]

"""
# Freetown 7 - Another central piece (same catalog_id)
# Image spec: 0% Clouds, Sun Azimuth 147.4825, Sun Elevation 58.164, Off Nadir Angle 21.444992
# Catalog ID 1040010047330800, WV03, taken 27 Jan 2019 at 11:56:02 AM
"""
freetown_7_id = '1040010047330800'
freetown_7_bbox = [-13.235023497945807, 8.45770608442553, -13.22290420466743, 8.470253729025437]

"""
# Bo 0 - Central Bo
# Image spec: 3% Clouds, Sun Azimuth 151.95882, Sun Elevation 55.083195, Off Nadir Angle 5.1485324
# Catalog ID 110400100472E6700, WV03, taken 1 Jan 2019 at 11:42:14 AM
"""
bo_0_id = '10400100472E6700'
bo_0_bbox = [-11.738960265392965, 7.9493544710931285, -11.725982664574985, 7.961407475207799]

"""
# Bo 1 - Southern Bo, same catalogid as central bo
# Image spec: 3% Clouds, Sun Azimuth 151.95882, Sun Elevation 55.083195, Off Nadir Angle 5.1485324
# Catalog ID 110400100472E6700, WV03, taken 1 Jan 2019 at 11:42:14 AM
"""
bo_1_id = '10400100472E6700'
bo_1_bbox = [-11.729690552019747, 7.920892082043292, -11.714103698795952, 7.936316847527625]

"""
# Bo 2 - Northern Bo, same catalogid as central bo
# Image spec: 3% Clouds, Sun Azimuth 151.95882, Sun Elevation 55.083195, Off Nadir Angle 5.1485324
# Catalog ID 110400100472E6700, WV03, taken 1 Jan 2019 at 11:42:14 AM
"""
bo_2_id = '10400100472E6700'
bo_2_bbox = [-11.739440918263428, 7.971161862754806, -11.721828461086263, 7.989172180709455]

"""
# Bo 3 - Small community near Bo
# Image spec: 0% Clouds, Sun Azimuth 144.718, Sun Elevation 54.362, Off Nadir Angle 27.919264
# Catalog ID 1040010038864E00, WV03, taken 14 Jan 2018 at 11:31:10 AM
"""
bo_3_id = '1040010038864E00'
bo_3_bbox = [-11.670458793214495, 7.829552047358135, -11.651026725376143, 7.8492032800338185]

"""
# Bo 4 - Western Bo, same image as Bo 2
# Image spec: 3% Clouds, Sun Azimuth 151.95882, Sun Elevation 55.083195, Off Nadir Angle 5.1485324
# Catalog ID 110400100472E6700, WV03, taken 1 Jan 2019 at 11:42:14 AM
"""
bo_4_id = '10400100472E6700'
bo_4_bbox = [-11.762220382952368, 7.9504424340376705, -11.748762130737305, 7.964129678551677]

"""
# Bo 5 - Eastern Bo, same image as Bo 2
# Image spec: 3% Clouds, Sun Azimuth 151.95882, Sun Elevation 55.083195, Off Nadir Angle 5.1485324
# Catalog ID 110400100472E6700, WV03, taken 1 Jan 2019 at 11:42:14 AM
"""
bo_5_id = '10400100472E6700'
bo_5_bbox = [-11.719493866294217, 7.940146054431308, -11.703907012415584, 7.954923183157281]

"""
# Kenema 0 - North Kenema
# Image spec: 1% Clouds, Sun Azimuth 149.603, Sun Elevation 55.476, Off Nadir Angle 29.914045
# Catalog ID 1030010074AF2E00, WV02, taken 6 Jan 2018 at 11:38:15 AM
"""
kenema_0_id = '1030010074AF2E00'
kenema_0_bbox = [-11.19012451381423, 7.89003967192376, -11.159088138956578, 7.92191244508875]

"""
# Kenema 1 - Central Kenema (same catalog id as kenema 0)
# Image spec: 1% Clouds, Sun Azimuth 149.603, Sun Elevation 55.476, Off Nadir Angle 29.914045
# Catalog ID 1030010074AF2E00, WV02, taken 6 Jan 2018 at 11:38:15 AM
"""
kenema_1_id = '1030010074AF2E00'
kenema_1_bbox = [-11.201248163706625, 7.857695057149697, -11.168014520080762, 7.889570339351238]

"""
# Kenema 2 - South Kenema (same catalog id as kenema 0)
# Image spec: 1% Clouds, Sun Azimuth 149.603, Sun Elevation 55.476, Off Nadir Angle 29.914045
# Catalog ID 1030010074AF2E00, WV02, taken 6 Jan 2018 at 11:38:15 AM
"""
kenema_2_id = '1030010074AF2E00'
kenema_2_bbox = [-11.218688961962473, 7.82183415970038, -11.18188476379146, 7.856300471456771]

"""
# Kenema 3 - Nearby community
# Image spec: 11% Clouds, Sun Azimuth 130.91626, Sun Elevation 61.27268, Off Nadir Angle 20.893147
# Catalog ID 10400100281CC300, WV03, taken 18 Feb 2017 at 11:32:02 AM
"""
kenema_3_id = '10400100281CC300'
kenema_3_bbox = [-11.349821090698244, 7.872383936969923, -11.334852217842128, 7.889376979332689]

"""
# Lemolo B 0 - Community where Anna worked, somewhat cloudy but most recent image (2019)
# Image spec: 6% Clouds, Sun Azimuth 134.07404, Sun Elevation 60.388767, Off Nadir Angle 26.539282
# Catalog ID 1040010046733F00, WV03, taken 19 Jan 2019 at 8:17:15 AM
"""
lomolo_b_0_id = '1040010046733F00'
lomolo_b_0_bbox = [36.03511333203642, -0.02089002638911957, 36.04761028080247, -0.0024065162165024107]

"""
# Lemolo B 1 - Community where Anna worked, no clouds image in 2017 
# Image spec: 0% Clouds, Sun Azimuth 77.82991, Sun Elevation 64.01895, Off Nadir Angle 23.785551
# Catalog ID 1030010067297800, WV02, taken 2 April 2017 at 8:57:19 AM
"""
lomolo_b_1_id = '1030010067297800'
lomolo_b_1_bbox = [36.03511333203642, -0.02089002638911957, 36.04761028080247, -0.0024065162165024107]

"""
# Echariria 0 - Community where Anna worked, most recent image
# Image spec: 0% Clouds, Sun Azimuth 125.474014, Sun Elevation 63.29197, Off Nadir Angle 12.499327
# Catalog ID 10400100467D0B00, WV03, taken 7 Feb 2019 at 8:20:04 AM
"""
echariria_0_id = '10400100467D0B00'
echariria_0_bbox = [36.21383857720503, -0.3578968717923196, 36.23114204485319, -0.34026082881621716]

"""
# Echariria 1 - Community where Anna worked, 2018 image with more green
# Image spec: 5% Clouds, Sun Azimuth 46.31241991, Sun Elevation 57.64345414, Off Nadir Angle 29.09273476
# Catalog ID 6bbcc455-1ec5-4533-9c59-7d4acc850a40-inv, WV04, taken 17 July 2018 at 9:02:51 AM
"""
echariria_1_id = '6bbcc455-1ec5-4533-9c59-7d4acc850a40-inv'
echariria_1_bbox = [36.21383857720503, -0.3578968717923196, 36.23114204485319, -0.34026082881621716]


def print_info(catalog_id, image, bbox):
    print("CatID: {} instantiated as a {} {}-band raster with {} rows and {} columns".format(catalog_id, image.dtype, *image.shape))
    print("  with geographic bounds: ({})".format(bbox))
    print("  in projection: {}".format(image.metadata["georef"]["spatialReferenceSystemCode"]))
    print("  at {:0.2f} meter resolution".format(image.metadata["image"]["groundSampleDistanceMeters"]))
    print("  and {:0.2f} degrees off nadir".format(image.metadata["image"]["offNadirAngle"]))
    print("  taken on {}".format(image.metadata["image"]["acquisitionDate"]))
    print("\n")


def grab_imagery(catalog_id, bbox, filename):
    # Grab multispectral and panchromatic images in bbox
    image = CatalogImage(catalog_id, band_type="MS", bbox=bbox)
    pan = CatalogImage(catalog_id, band_type="Pan", bbox=bbox)
    # Print multispectral and panchromatic image info
    print_info(catalog_id, image, bbox)
    print_info(catalog_id, pan, bbox)
    # Save multispectral and panchromatic images
    image.geotiff(path="imagery/" + filename + "_ms.tif", proj="EPSG:4326")
    pan.geotiff(path="imagery/" + filename + "_pan.tif", proj="EPSG:4326")


# #### RUN ##### #

# Get the imagery you want
grab_imagery(echariria_1_id, echariria_1_bbox, "echariria_1")

