# Test code for printing annotation clusters for a specific image
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np

# This script plots annotation clusters for a particular subject_id for demonstration

data_path = Path("C:/Users/alyci/PycharmProjects/GeoDESA/zooniverse/datasamples")
subject_id = "40169529"
image_name = "Alikalia_20180403_PS_RGB_Crop1_Rend_02_04.png"
image_path = data_path / image_name

# Raw data for this image
r_x = [123.96875, 76.96875, 59.96875, 63.96875, 72.96875, 123.96875, 419.96875, 86.07499694824219, 130.87498474121094, 64.47502136230469, 117.84369659423828, 66.697509765625, 51.13749694824219, 78.4375, 128.4375, 62.4375, 81.3671875, 65.265625, 129.65625, 362.96875, 59.96875, 116.96875, 69.96875, 81.70162512434766, 65.96875, 128.2197196187625, 74.14583587646484, 59.47916793823242, 120.1458511352539, 127.37277221679688, 80.96623229980469, 65.12008666992188]
r_y = [64.0, 135.0, 348.0, 356.5, 139.5, 65.50001525878906, -11.499996185302734, 141.40003967285156, 67.80001068115234, 356.6000061035156, 61.549896240234375, 132.8134002685547, 355.136962890625, 140.49996948242188, 65.5, 353.5, 147.5234375, 356.765625, 63.2109375, 0.0, 355.0, 64.0, 140.0, 143.60942452042116, 356.125, 66.96539783737978, 138.33335876464844, 351.6666564941406, 59.66667556762695, 70.38327026367188, 143.03024291992188, 359.2168273925781]
r_width = [45.0, 73.0, 35.0, 29.0, 78.0, 44.0, 76.0, 66.4000244140625, 36.0, 27.199981689453125, 52.556785583496094, 88.68113708496094, 48.49425506591797, 65.0, 37.0, 32.00001525878906, 58.94921875, 28.31640625, 42.01171875, 62.0, 33.0, 61.0, 87.0, 64.53424975130469, 29.0, 41.49806076247505, 74.66667938232422, 35.33333969116211, 56.00000762939453, 33.95599365234375, 59.98893737792969, 23.769210815429688]
r_height = [52.0, 75.0, 59.0, 39.0, 59.000030517578125, 40.99998474121094, 46.000003814697266, 52.79997253417969, 40.0, 40.0, 48.207801818847656, 76.61943054199219, 35.87261962890625, 60.0, 38.000030517578125, 45.0, 47.64453125, 39.20703125, 46.76953125, 25.0, 42.0, 46.0, 56.0, 47.031150959157635, 41.0, 38.31920432524042, 62.0, 51.333343505859375, 57.33333206176758, 33.95600891113281, 46.40655517578125, 33.956024169921875]
r_angle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.663229308632125, 21.801432876294257, 0.0, 0.0, 0.0, 0.0, 14.651352978546877, 23.466064464496657, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.577970467567285, 0.0, 17.198541220065866, 0.0, 0.0, 0.0, 18.546281496513433, 14.958639728652685, 0.0]

# Cluster data for this image
c_x = [61.77898979187012, 124.74570243062627, 76.92971375144843]
c_y = [354.8511077880859, 64.85762033488544, 140.17098731288198]
c_width = [32.111320877075194, 44.90225663777095, 71.6220246760582]
c_height = [42.63690185546875, 44.15858936257776, 58.250167097868896]
c_angle = [0.0, 8.101, 5.385]

# Load the image for the defined subject_id
im = np.array(Image.open(image_path), dtype=np.uint8)

# Create figure and axes
fig, ax = plt.subplots(1)
plt.axis('off')

# Display the image
ax.imshow(im)

# Print all raw rectangles on the image
for i in range(len(r_x)):
    # Create a Rectangle patch corresponding to the cluster
    rect = patches.Rectangle((r_x[i], r_y[i]), r_width[i], r_height[i], r_angle[i], linewidth=1, edgecolor='yellow', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

# Print all clusters on the image
for i in range(len(c_x)):
    # Create a Rectangle patch corresponding to the cluster
    rect = patches.Rectangle((c_x[i], c_y[i]), c_width[i], c_height[i], c_angle[i], linewidth=2, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

# Show the image
plt.show()


#subject_set_path = Path("C:/Users/alyci/Documents/DPhil/GeoDESA/Zooniverse/SubjectSets/Lomolo_B_20180109_SubjectSet")
#data_path = Path("C:/Users/alyci/Documents/DPhil\GeoDESA\Zooniverse\DataExports\DataExport_20200922\panoptes\All")




# Rotate the patch as needed
# t_start = ax.transData
# t = transforms.Affine2D().rotate_deg(-45)
# t_end = t_start + t
# rect.set_transform(t_end)



