#This program uses a pretrained model to detect faces and draws a boxes around them
#Cropped images of the detected face can be saved
#This code is based off the following reference: https://www.pyimagesearch.com/2021/04/05/opencv-face-detection-with-haar-cascades/

#imports
from imutils.video import VideoStream

import imutils
import cv2
import os

picCount = 0 #counter for images
# load the haar cascade face detector from
print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
cwd = os.getcwd()

print("Frame - Press s to capture/save image and q to quit")
print("Please make sure box is around face before saving image!!!")
# loop over the frames from the video stream

try:
	os.mkdir("data")

except FileExistsError:
	pass

try:
	os.mkdir("data/person")
except FileExistsError:
	pass

while True:
	# grab the frame from the video stream, resize it, and convert it
	# to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=700)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# perform face detection
	rects = detector.detectMultiScale(gray, scaleFactor=1.05,minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    # loop over the bounding boxes
	original_frame = vs.read()
	for (x, y, w, h) in rects:
		# draw the face bounding box on the image
		cv2.rectangle(frame, (x-10, y-10), (x + w+10, y + h+10), (0, 255, 0), 3)
		crop = original_frame[y-10:y+h+10, x-10:x+w+10]#crop face
	# show the output frame
	cv2.imshow("Frame - Press s to capture/save image and q to quit", frame) #Show frame
	key = cv2.waitKey(1) & 0xFF #key listener
	if key == ord("s"):
		fname = "{}/{}/saved_img.jpg" #create name for image(not unique and will overwrite)
		cv2.imwrite(filename=fname.format(cwd,"data/person"), img=crop)
		print("Image Saved")# if the `s` key was pressed, a cropped image of the detected face will be saved
		print("Running data augmentations")
		break

	elif key == ord("q"):
		print("Quitting")# if the `q` key was pressed, break from the loop
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
