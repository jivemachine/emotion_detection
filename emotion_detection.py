#USAGE: python face_detection.py

# imports
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import image_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
