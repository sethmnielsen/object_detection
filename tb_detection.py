#!/usr/bin/env python
# # This script detects tennis balls in a video feed.
# Be sure to follow the [installation instructions]
# (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
# before you start.
from __future__ import division
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

sys.path.append("..")

# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util

# # Model preparation
# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
MODEL_NAME = 'tennisball_graph5'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph_inception.pb'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph_inception5-3.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'tb_label_map.pbtxt')
NUM_CLASSES = 1

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# # Detection
# Choose video feed to be a video file (e.g. 'path/to/file') or a camera input (e.g. 0 or 1)
# cap = cv2.VideoCapture(os.path.expanduser('~/Videos/vid3.mp4'))
cap = cv2.VideoCapture(os.path.expanduser('~/Videos/urc_autonomy/A4.MOV'))
# cap = cv2.VideoCapture('/dev/videoZED')
w = 1920
h = 1080
cap.set(3, w)
cap.set(4, h)
prev_score = 0

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    while cap.isOpened():
        for i in range(0):
            _ = cap.grab()
        ret, frame = cap.read()
        # image_np = frame # use this line for any video feed that isn't the ZED
        image_np = frame[0:h, 0:w] # for using only the left camera feed if using ZED as input

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        # vis_util.visualize_boxes_and_labels_on_image_array(
          # image_np,
          # np.squeeze(boxes),
          # np.squeeze(classes).astype(np.int32),
          # np.squeeze(scores),
          # category_index,
          # use_normalized_coordinates=True,
          # line_thickness=8)
        box = boxes[0][0]
        im_h, im_w, _ = image_np.shape
        # corners is [x, y, x_max, y_max] (y-value increases downwards)
        corners = (int(box[1]*im_w), int(box[0]*im_h), int(box[3]*im_w), int(box[2]*im_h))
        x = corners[0]
        y = corners[1]
        x_max = corners[2]
        y_max = corners[3]
        length = x_max-x
        height = y_max-y
        raw_score = scores[0][0]
        squareness = 1 - abs(length-height) / (length+height)
        if squareness < 0:
            raw_score = 0

        alpha = 0.1
        filt_score = (1-alpha)*prev_score + alpha*raw_score
        prev_score = filt_score

        if filt_score > .50:
                cv2.putText(image_np,'{:2.2f}'.format(filt_score),(x, y-5),cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,255), 2)
                cv2.rectangle(image_np, (x, y), (x_max, y_max), (0,0,255), 2)
        elif raw_score > .50:
                cv2.rectangle(image_np, (x, y), (x_max, y_max), (255,0,0), 2)
        else:
            corners = (-1, -1, -1, -1)
        # print str(corners) + ', {:1.2f}'.format(filt_score)
        # cv2.imshow('object detection 3', cv2.resize(image_np, (640, 480)))
        cv2.imshow('object detection 2', image_np)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break

    cv2.destroyAllWindows()
    cap.release()
