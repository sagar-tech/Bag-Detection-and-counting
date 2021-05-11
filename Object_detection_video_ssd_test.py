######## Video Object Detection Using Tensorflow-trained Classifier #########

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import datetime
from datetime import date
from tkinter import font



# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssd_model'
VIDEO_NAME = 'video2.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'ssd_model','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
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

# Initialize Tracker
tracker = cv2.TrackerCSRT_create()
trackers = cv2.MultiTracker_create()
xmin_l,xmax_l,ymin_l,ymax_l = 0,0,350,0

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)


start_time = datetime.datetime.now()
num_frames = 0

count=0
while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)
    height,width = frame.shape[0:2]
    frame[0:70,0:width]=[0,0,255]
    print("h,w",height,width)

    score_thresh = 0.80

    Orientation= 'bt'

    Line_Perc1=float(60)

    num_box_detect = 6

    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    (success, boxes) = trackers.update(frame)
    # loop over the bounding boxes and draw then on the frame
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # cv2.line(frame,(0,245),(1000,245),(0,0,255),2)#Red line

    cv2.putText(frame,'Count:{}'.format(count),(450,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),1)

    # Draw the results of the detection (aka 'visulaize the results')
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     frame,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=1,
    #     min_score_thresh=0.80)


    box=np.squeeze(boxes)
    # print(len(box))
    scores_idx = np.where(scores > 0.80)
    scores = scores[scores_idx]
    classes = classes[scores_idx]
    boxes = boxes[scores_idx]
    # print("boxes::::", boxes)
    n_box = len(boxes)
    if n_box >0:
        
        ymin = (int(box[0,0]*height))
        xmin = (int(box[0,1]*width))
        ymax = (int(box[0,2]*height))
        xmax = (int(box[0,3]*width))

        # print("ymin,xmin,ymax,xmax",ymin,xmin,ymax,xmax)
        if ymin < ymin_l:
            trackers.add(tracker, frame, (xmin,ymin,xmax,ymax))
            ymin_l = ymin
            ymax_l = ymax
            xmin_l = xmin
            xmax_l = xmax

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
#cv2.destroyAllWindows()
