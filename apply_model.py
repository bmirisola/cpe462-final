import os
import cv2

person_model = os.path.join(os.getcwd(), "model/face_classifier.h5")
face_detector = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
