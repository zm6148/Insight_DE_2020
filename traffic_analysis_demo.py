import functions as fun
import os
import cv2
import tensorflow as tf
import glob
import numpy as np

# current script path
scripty_path = os.path.dirname(os.path.realpath(__file__))

# config yolo model
config_path = scripty_path + '/yolo_model/yolov3.cfg'
weight_path = scripty_path + '/yolo_model/yolov3.weights'
class_path = scripty_path + '/yolo_model/yolov3.txt'
# build net
net = cv2.dnn.readNet(weight_path, config_path)

# define classes
classes = None
with open(class_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
# pre-define color scheme
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# load in .mp4 and find all frames
# video_name = "/The Good, the Bad and the Ugly - The Final Duel (1966 HD).mp4"
video_name = "/traffic.mp4"
frame_path = scripty_path + "/frames"
vidcap = cv2.VideoCapture(scripty_path + video_name)
total_frames = vidcap.get(7)
all_frames = range(0, int(total_frames))

while(True):
    # Capture frame-by-frame
    ret, frame = vidcap.read()

    # Our operations on the frame come here
    image_o, class_ids = fun.object_identification(frame, classes, net, COLORS)
    print('Cars: {}'.format(class_ids.count(2)))
    print('Trucks: {}'.format(class_ids.count(7)))
    
    # Display the resulting frame
    cv2.imshow('frame',image_o)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    