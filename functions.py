import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from PIL import Image
import urllib

################################################################################
# function to encode faces
# take in the path of faces you want to detect
def get_encoded_faces(known_faces_path):
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk(known_faces_path):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file(known_faces_path + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded

################################################################################
# function to classify faces based on the faces encoded using
# photos of persons of interest
# im is path for now
def classify_face(img, known_faces):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: return found faces, altered img, and wether there is a match 
    """
    #faces = get_encoded_faces(known_faces_path)
    faces_encoded = list(known_faces.values())
    known_face_names = list(known_faces.keys())

    #img = cv2.imread(im, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)
    
    # check wether target face was found
    common_name = list(set(face_names).intersection(known_face_names))
    if not common_name:
        match = 0
    else:
        match = 1
    # return found faces, altered img, and wether there is a match   
    return face_names, img, match
    
################################################################################
# functiont to check wether this frame of img contains any weapon
# now: knife, short gun, long gun
# im is path for now
def find_weapon(img_cv, model):
    
    m,n = 50,50
    
    weapon = ['knife', 'gun', 'gun']
    #model = tf.keras.models.load_model(model_path)
    x=[]
    # img_cv = cv2.imread(im, 1)
    # convert to pillow image
    img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    imrs = im_pil.resize((m,n))
    imrs=img_to_array(imrs)/255;
    imrs=imrs.transpose(2,0,1);
    imrs=imrs.reshape(3,m,n);
    x.append(imrs)
        
    x=np.array(x);
    predictions = model.predict(x)
    greatest_p_index = np.argmax(predictions)
    
    if np.amax(predictions) > 0.9:
        return weapon[greatest_p_index], 1
    else:
        return 'no weapon', 0
        
################################################################################
# functions for object identification
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, confidence, x, y, x_plus_w, y_plus_h, label, color):

    #label = str(classes[class_id])

    #color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (color), 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    
def object_identification(image, classes, net, COLORS):
    
    #image = cv2.imread(input_image_path)
    Width = image.shape[1]
    Height = image.shape[0]
    
    # classes = None
    # with open(class_path, 'r') as f:
    #     classes = [line.strip() for line in f.readlines()]
    
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.65:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        label = str(classes[class_ids[i]])
        color = COLORS[class_ids[i]]
        draw_prediction(image, confidences[i], int(x), int(y), int(x+w), int(y+h), label, color)
    
    return image, class_ids
    

