######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py
## but I changed it to make it more understandable to me.

## Edited by Alycia Leonard (2019-05-16) - Generic version to be left in top-level object_detection folder
## Edited again by Alycia lol (2021-08-12) - go through all files in a file list CSV, save results to folder.

# RUN THIS ON THE BEAST. It's there.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pandas as pd

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

## CHECK AND DEFINE THESE PARAMETERS ##

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model used for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

# CSV of list of samples
SAMPLE_LIST = os.path.join(CWD_PATH,'images','test_images_20210804_2.csv')

# Script starts from here

# Load the label map.
# Label maps map indices to category names (i.e. when our convolution
# network predicts `5`, we know that this corresponds to `king`)
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Read in list of files
filenames = pd.read_csv(SAMPLE_LIST)
filenames_list = filenames['filenames'].to_list()

# For each file in the list:
for i in filenames_list:
    # Name of the sample we are running detection on (no extension)
    SAMPLE_NAME = i
    # Path to image
    PATH_TO_IMAGE = os.path.join(CWD_PATH,'images','test',SAMPLE_NAME+'.png')
    print(PATH_TO_IMAGE)

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Reshape the boxes result from (1, X, Y) to (X, Y)
    bshape = boxes.shape
    boxes2 = np.reshape(boxes, (bshape[1], bshape[2]))

    # Check for reshape: Print boxes contents as list to saved boxes2 CSV (uncomment if desired)
    # print(boxes.tolist())

    # Print the shape of all results
    # print('Size of "boxes" results:', bshape)
    # print('Size of "boxes2" results (reshaped "boxes"):', boxes2.shape)
    # print('Size of "scores" results:', scores.shape)
    # print('Size of "classes" results:', classes.shape)
    # print('Size of "num" results:', num.shape)

    # Save the boxes, scores, classes, num for this run
    #SAVE_PATH = os.path.join(CWD_PATH,'eval','test_detections',)
    #np.save(os.path.join(CWD_PATH,'eval','test_detections',SAMPLE_NAME+'_boxes'), boxes)
    #np.savetxt(os.path.join(CWD_PATH,'eval','test_detections',SAMPLE_NAME+'_scores.csv'), scores, delimiter=',')
    #np.savetxt(os.path.join(CWD_PATH,'eval','test_detections',SAMPLE_NAME+'_classes.csv'), classes, delimiter=',')
    #np.savetxt(os.path.join(CWD_PATH,'eval','test_detections',SAMPLE_NAME+'_num.csv'), num, delimiter=',')
    #np.savetxt(os.path.join(CWD_PATH,'eval','test_detections',SAMPLE_NAME+'_boxes2.csv'), boxes2, delimiter=',')

    # Draw the results of the detection on the image (aka visualize the results)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.50,
        skip_scores=False,
        skip_labels=True
        )

    # Display the image and save it
    # cv2.imshow('Object detector', image)
    cv2.imwrite(os.path.join(CWD_PATH,'eval','test_detections_2', SAMPLE_NAME+'_results.png'), image)
    # print('Displayed image has been saved!')

    # Press any key to close the image
    #cv2.waitKey(0)

    # Clean up
    cv2.destroyAllWindows()
