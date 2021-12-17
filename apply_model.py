import os
import cv2
import keras.models
import numpy as np
from imutils.video import VideoStream
import tensorflow as tf

img_path = "data/person/saved_img.jpg"
img = cv2.imread(img_path)
input_shape = img.shape
img_height = img.shape[0]
img_width = img.shape[1]

def get_extended_image(img, x, y, w, h, k=0.1):
    '''
    Function, that return cropped image from 'img'
    If k=0 returns image, cropped from (x, y) (top left) to (x+w, y+h) (bottom right)
    If k!=0 returns image, cropped from (x-k*w, y-k*h) to (x+k*w, y+(1+k)*h)
    After getting the desired image resize it to 250x250.
    And converts to tensor with shape (1, 250, 250, 3)

    Parameters:
        img (array-like, 2D): The original image
        x (int): x coordinate of the upper-left corner
        y (int): y coordinate of the upper-left corner
        w (int): Width of the desired image
        h (int): Height of the desired image
        k (float): The coefficient of expansion of the image

    Returns:
        image (tensor with shape (1, img_height, img_width, 3)
    '''

    with tf.device('/CPU:0'):

        # The next code block checks that coordinates will be non-negative
        # (in case if desired image is located in top left corner)
        if x - k*w > 0:
            start_x = int(x - k*w)
        else:
            start_x = x
        if y - k*h > 0:
            start_y = int(y - k*h)
        else:
            start_y = y

        end_x = int(x + (1 + k)*w)
        end_y = int(y + (1 + k)*h)

        face_image = img[start_y:end_y,
                         start_x:end_x]
        face_image = tf.image.resize(face_image, [img_height,img_width])
        # shape from (img_height, img_width, 3) to (1, img_height, img_width, 3)
        face_image = np.expand_dims(face_image, axis=0)
        return face_image

print("\n" + "RUNNING MODEL")
person_identifier_model = os.path.join("model/face_classifier.h5")
person_identifier = keras.models.load_model(person_identifier_model)
face_detector = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
name = 'Benny'
vs = VideoStream(src=0).start()
while True:
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in rects:
        with tf.device('/CPU:0'):
            face_image = get_extended_image(frame, x, y, w, h, 0.5)
            result = person_identifier.predict(face_image)
            confidence = np.array(result[0]).max(axis=0)
        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 3)
        cv2.putText(frame, "{:6} - {:.2f}%".format(name,confidence * 100), (x, y), cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)  # thickness in px
    cv2.imshow("Prediction with Confidence", frame)

    key = cv2.waitKey(1) & 0xFF  # key listener
    if key == ord("q"):
        print("Quitting")  # if the `q` key was pressed, break from the loop
        break

