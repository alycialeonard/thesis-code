# Image processing and plots to help Nisrine with RELCON
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import cv2
from matplotlib import pyplot as plt
import numpy as np

map_path = "C:\\Users\\alyci\\Documents\\DPhil\\RELCON\\map.bmp"

# Read the three-channel image
img = cv2.imread(map_path)
# Convert to one-channel grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image
ret, thresh1 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

# Show the image
plt.imshow(thresh1, 'gray')
plt.title('Binary image')
plt.xticks([])
plt.yticks([])
plt.show()

# Save the image
cv2.imwrite('C:\\Users\\alyci\\Documents\\DPhil\\RELCON\\map_binary.bmp', thresh1)

# Resize the image
print('Original Dimensions: ', img.shape)
scale_percent = 300  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(thresh1, dim)

# Save the resized image
cv2.imwrite('C:\\Users\\alyci\\Documents\\DPhil\\RELCON\\map_binary_resized.bmp', resized)

# Perform Canny edge detection on the resized binary image
edges = cv2.Canny(resized, 50, 150, apertureSize=3)

# Show Canny edges
plt.imshow(edges, 'gray')
plt.title('Canny Edges')
plt.xticks([])
plt.yticks([])
plt.show()

# Save pure Canny edges
cv2.imwrite('C:\\Users\\alyci\\Documents\\DPhil\\RELCON\\map_cannyedges.bmp', edges)

# Define kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

# Dilate edges then open and close to get rid of any artifacts
dilated = cv2.dilate(edges, kernel, iterations=1)
opened = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)

# Show dilated Canny edges
plt.imshow(opened, 'gray')
plt.title('Edges after dilation (morphological)')
plt.xticks([])
plt.yticks([])
plt.show()

# Save the dilated Canny edges image
cv2.imwrite('C:\\Users\\alyci\\Documents\\DPhil\\RELCON\\map_dilatededges.bmp', dilated)
