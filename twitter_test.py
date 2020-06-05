import tweepy
import functions as fun
import os
import numpy as np
import cv2
import urllib

# current script path
scripty_path = os.path.dirname(os.path.realpath(__file__))

# config yolo model
config_path = scripty_path + '/yolo_model/yolov3.cfg'
weight_path = scripty_path + '/yolo_model/yolov3.weights'
class_path = scripty_path + '/yolo_model/yolov3.txt'
# build net
net = cv2.dnn.readNet(weight_path, config_path)
# build output layer
output_layers = fun.get_output_layers(net)

# config face
# read in target face, known faces, and trained weapon detection model
known_face_path = scripty_path + "/known_faces/"
# encode known faces
known_faces = fun.get_encoded_faces(known_face_path)

# define classes
classes = None
with open(class_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
# pre-define color scheme
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

auth = tweepy.OAuthHandler('vwxlnP5SclOUAyE4JqgDMwkxx', 'FaPP7mPYHRhL4Drii8wvw6Qbcj3SUcG3ImDCB8RqkVof9F55cD')
auth.set_access_token('945213054-GEDcrnXhEwsUdC39H1LUIv5VUXOhTgEyCpi2gUXt', 'hRsPy8LNf1sfsoUggwtFZzDpIHyGcOdhv1Dx5b9xi5emp')
api = tweepy.API(auth)


tweets = tweepy.Cursor(api.search,
                           q="protest",
                           count=100,
                           result_type="recent",
                           include_entities=True,
                           lang="en").items(100)
   
media_files = set()
for status in tweets:
    media = status.entities.get('media', [])
    if(len(media) > 0):
        media_files.add(media[0]['media_url'])
        
urls = list(media_files)
url = urls[0]
url_response = urllib.urlopen(url)
image = np.asarray(bytearray(url_response.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
# object
image_o, class_ids = fun.object_identification(image, classes, net, output_layers, COLORS)
# face
face_names, image_f, match_face = fun.classify_face(image_o, known_faces)

cv2.imshow("object detection", image_f)