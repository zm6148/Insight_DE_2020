import functions as fun
import os
import cv2
import tensorflow as tf
import glob

# current script path
scripty_path = os.path.dirname(os.path.realpath(__file__))

# read in target face, known faces, and trained weapon detection model
known_face_path = scripty_path + "/known_faces/"
model_path = scripty_path + "/weapon_model/model_latest.h5"
result_path = scripty_path +  "/results/"
model = tf.keras.models.load_model(model_path)
# encode known faces
known_faces = fun.get_encoded_faces(known_face_path)

# load in .mp4 and find all frames
# video_name = "/The Good, the Bad and the Ugly - The Final Duel (1966 HD).mp4"
video_name = "/Dirty Harry.mp4"
frame_path = scripty_path + "/frames"
vidcap = cv2.VideoCapture(scripty_path + video_name)
total_frames = vidcap.get(7)
all_frames = range(0, int(total_frames))


for ii in range(0, int(total_frames),10):
    vidcap.set(1, ii)
    ret, frame = vidcap.read()
    print(ii)
    
    # check weapon first
    weapon, match_weapon = fun.find_weapon(frame, model)
    if match_weapon == 1:
        print("weapon match")
        face_names, img, match_face = fun.classify_face(frame, known_faces)
    else:
        continue
          
    # change start and end index
    if  match_face == 1:#if yes return
        print("face match")
        cv2.imwrite(result_path + "%d.jpg"%ii, img) 
        
################################################################################
img_array = []
for filename in glob.glob(result_path + '*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
out = cv2.VideoWriter(scripty_path + '/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


# binary search on all frames
# first = 0
# last = len(all_frames) - 1
# found = False
# # loop to find which frame is a match for target face and weapon
# while (first < last and not found):
#     # starting point
#     mid = int((first + last)/2)
#     print(mid)
#     
#     # load the frame at mid frame
#     vidcap.set(1, mid)
#     ret, frame = vidcap.read()
#     
#     # chech if it is the frame
#     weapon, match_weapon = fun.find_weapon(frame, model_path)
#     if match_weapon == 1:
#         face_names, img, match_face = fun.classify_face(frame, known_faces)
#     else:
#         continue
#          
#     # change start and end index
#     if  known_faces == 1:#if yes return
#         found = True
#         break
#     else:#no, change first and last point
#         first = mid 
#         last = len(all_frames) - 1   
#     
# cv2.imshow('Video', img)
# print(weapon)
