# Get data for classifier
# Goal: Classify tiles as interesting (i.e. potential settlements) or not interesting (definitely no settlements)
# Heavily references https://towardsdatascience.com/all-the-steps-to-build-your-first-image-classifier-with-code-cf244b015799
# Author: Alycia Leonard, Energy and Power Group, University of Oxford
# Contact: alycia.leonard@eng.ox.ac.uk

import numpy as np
import os
import cv2
import random
import pickle

# Directory where training data is found
DATADIR = "C:\\Users\\alyci\\Documents\\DATA\\earthi\\Echariria_Classify_Example"

# All the categories you want your neural network to detect
CATEGORIES = ["Interest", "Reject"]

# The size of the images that your neural network will use
IMG_SIZE = 256

data = []

# Checking or all images in the data folder
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append([new_array, class_num])
        except Exception as e:
            pass

# Shuffle the data (images + labels)
random.shuffle(data)

# Create placeholders for feature (X) and label (y) lists
X = []
y = []

# Append features (images) and labels to separate lists
for features, label in data:
    X.append(features)
    y.append(label)

# Make features an array and check shape/length for match
X = np.array(X)
print("Shape of features:", X.shape)
print("Length of labels:", len(y))

# Creating the files containing all the information about your data
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

