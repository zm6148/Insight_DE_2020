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

class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        media = status.entities.get('media', [])
        if(len(media) > 0):
            url = media[0]['media_url']
            print(url)
            print(status.text)
            print(status.coordinates)
            print(status.place)
            print(status.geo)
            # Our operations on the frame come here
            url_response = urllib.urlopen(url)
            image = np.asarray(bytearray(url_response.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            # object
            image_o, class_ids = fun.object_identification(image, classes, net, COLORS)
            # face
            face_names, image_f, match_face = fun.classify_face(image_o, known_faces)
    
            # Display the resulting frame
            cv2.imshow('frame',image_o)
            cv2.waitKey(1)
        
    
    def on_error(self, status_code):
        if status_code == 420:
            return False

        
myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
myStream.filter(track=['covid-19'])

# class StreamListener(tweepy.StreamListener):
#     def on_status(self, status):
#         
#         if status.favorite_count is None or status.favorite_count < 10:
#             return
#             
#         description = status.user.description
#         loc = status.user.location
#         text = status.text
#         coords = status.coordinates
#         name = status.user.screen_name
#         user_created = status.user.created_at
#         followers = status.user.followers_count
#         id_str = status.id_str
#         created = status.created_at
#         retweets = status.retweet_count
#         bg_color = status.user.profile_background_color
#         media = status.entities.get('media', [])
#         print(id_str)
#         
#     def on_error(self, status_code):
#         if status_code == 420:
#             return False
# 
# stream_listener = StreamListener()
# stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
# stream.filter(track=["protest"])