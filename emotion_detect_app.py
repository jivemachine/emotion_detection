# imports
import keras
from keras.models import load_model
from time import sleep
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
classifier = load_model("./Emotion_Detection.h5")
class_labels = ["Angry", "Happy", "Neutral", "Sad", "Suprised"]

class DetectEmotion(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
    
    def __del__(self):
        self.cap.release()
    
    def get_frame(self):
        # grab single frame of video
        ret, frame = self.cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.5, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                global sess
                global graph
                with graph.as_default():
                    set_session(sess)
                    preds = classifier.predict(roi)[0]
                    print("Get it")
                    label = class_labels[preds.argmax()]
                    label_position=(x,y)
            
        
        
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_TRIPLEX, 2,(0,255,0),3)
            else:
                cv2.putText(frame, "No Face Found", (20,60), cv2.FONT_HERSHEY_TRIPLEX,2(0,255,0), 3)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return (jpg.tobytes)
