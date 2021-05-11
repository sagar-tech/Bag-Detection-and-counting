######## Video Object Detection Using Tensorflow-trained Classifier #########

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import orien_lines
import datetime
from datetime import date
import count
import detector_utils
from tkinter import font



# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssd_model'
VIDEO_NAME = '1.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training_ssd','labelmap.pbtxt')

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

    
    #cv2.putText(frame,'Bag Count:',(10,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)

    #cv2.line(frame,(0,height-125),(640,height-125),(0,255,255),2)
    #cv2.rectangle(frame, (352,342), (589,227), (0,0,255), 3) 
    #print("line cordinates:::::::",(250,height-125), (400,height-125))
    '''fgMask=cv2.absdiff(frame1,frame2)
    fgMask=cv2.cvtColor(fgMask,cv2.COLOR_BGR2GRAY)
    _,thres = cv2.threshold(fgMask,50,255,cv2.THRESH_BINARY)
    frame1=frame2'''

    #cv2.line(frame,(220,200),(1000,200),(0,0,255),2)

    #conts,_=cv2.findContours(thres,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    '''for c in conts:
        if cv2.contourArea(c) < 300:
            continue
        x,y,w,h =cv2.boundingRect(c)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        xMid=int((x+(x+w))/2)
        yMid=int((y+(y+h))/2)
        cv2.circle(frame,(xMid,yMid),5,(0,0,255),5)

        if yMid == 200:
            count+=1'''
    
                 
                        
    #cv2.putText(frame,'Count:{}'.format(count),(450,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
    
    


    

    score_thresh = 0.80

    

    Orientation= 'bt'

    Line_Perc1=float(60)

    num_box_detect = 6

            

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    

    #Line_Position1=orien_lines.drawsafelines(frame,Orientation,Line_Perc1)

    '''a=detector_utils.draw_box_on_image(
                num_box_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame,Line_Position1,Orientation)
    lst1.append(a)'''

    '''num_frames += 1
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time

    orien_lines.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)'''


    '''for (x,y,w,h) in np.squeeze(boxes):
        bagCy=int(y+h/2)
        lineCy=height-125

       
        if yMid == 125:
            count+=1

        if (bagCy<lineCy+6 and bagCy>lineCy-6):

            count=count+1'''
        #cv2.putText(frame,str(count),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)    

    cv2.line(frame,(0,245),(1000,245),(0,0,255),2)#Red line
    #cv2.line(frame,(0,190),(1000,190),(0,255,0),1)#Green line
    #cv2.line(frame,(0,210),(1000,210),(0,255,0),1)#Green line

    #print("line cordinates:::::::",(0,1000), (260,260))

    
    

    
    

    #orien_lines.draw_text_on_image("Count : " + str("{0:.2f}".format(count)), frame)        
      
    '''for (xmin,ymin,xmax,ymax) in np.squeeze(boxes):

        (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
    box=np.squeeze(boxes)
    for i in range(len(boxes)):
        ymin = (int(box[i,0]*height))
        xmin = (int(box[i,0]*width))
        ymax = (int(box[i,0]*height))
        xmax = (int(box[i,0]*width))



    (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
    bagCy=int(ymin+height/2)
    lineCy=height-125

    xMid=int((xmin+(xmin+width))/2)
    yMid=int((ymin+(ymin+height))/2)
    print("yMId",yMid)
    
    print("cordinates top bottom left::::",  (left+ right)/2, (top+bottom)/2)
    x1, x2, x3 = 250, 400, (left+ right)/2
    y1, y2, y3 = 227, 227, (top+bottom)/2
    if (y3 == y2) and (x1 <= x3 <= x2):
        print("ceil value of top::::", x1,x2,x3,y1,y2,y3)
        count += 1
        break
    if yMid <= 250:
        count+=1'''

    #if (bagCy<lineCy+6 and bagCy>lineCy-6):


    

        #print("main::::::: ",ymin,xmin,ymax,xmax)
            

            
        
    '''p1 = (int(left), int(top))
    p2 = (int(right), int(bottom))
    if(Orientation=="bt"):
     
        bounding_mid = (int((p1[0]+p2[0])/2),int(p1[1]))
    
        if(bounding_mid):
            cv2.line(img = frame, pt1=bounding_mid, pt2=(bounding_mid[0],Line_Position1), color=(255, 0, 0), thickness=1, lineType=8, shift=0)
            distance_from_line = bounding_mid[1]-Line_Position1
   
    
    
    if (distance_from_line <= 0) :
            
             count+=1'''

    cv2.putText(frame,'Count:{}'.format(count),(450,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)     

    #a=count.drawboxtosafeline(frame,p1,p2,Line_Position1,Orientation)
    #cv2.putText(frame,str(a),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)      
          

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=1,
        min_score_thresh=0.80)


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

        print("ymin,xmin,ymax,xmax",ymin,xmin,ymax,xmax)    

        #if xmin > 250 and xmin < 400 and ymin:
            #cv2.rectangle(frame,(xmin,ymin),(xmin+width,ymin+height),(0,255,0),2)
        xMid=int((xmin+xmax)/2)
        yMid=int((ymin+ymax)/2)
        cv2.circle(frame,(xMid,yMid),5,(0,0,255),5)
        print("xMid,yMid:::",xMid,yMid)
        if yMid >= 245 and yMid<=248 and xMid >=245 and xMid <= 400:
            count=count+1
            #break


        '''distance=yMid-275
        
        if distance <= 0 and yMid >= 275 and yMid<=277:
            count=count+1'''



    '''x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = np.stack((cx, cy, w, h), axis=-1)

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    #cv2.circle(frame,(cx,cy),5,(0,0,255),5)
    if cx >= 190 and cx <= 210:
        count=count+1'''




    



    

    '''total_passed_vehicle = total_passed_vehicle + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Vehicles: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )

                # when the vehicle passed over line and counted, make the color of ROI line green
                if counter == 1:
                    cv2.line(input_frame, (0, 200), (640, 200), (0, 0xFF, 0), 5)
                else:
                    cv2.line(input_frame, (0, 200), (640, 200), (0, 0, 0xFF), 5)

                # insert information text to video frame
                cv2.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)'''
                
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
#cv2.destroyAllWindows()
