from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
 #construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str,default="haarcascade_frontalface_default.xml")
args = vars(ap.parse_args())
# load the haar cascade face detector from
print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier(args["cascade"])
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# loop over the frames from the video stream
while True:
	# grab the frame from the video stream, resize it, and convert it
	# to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=700)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# perform face detection
	rects = detector.detectMultiScale(gray, scaleFactor=1.05,minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    # loop over the bounding boxes
	for (x, y, w, h) in rects:
		# draw the face bounding box on the image
		cv2.rectangle(frame, (x-10, y-10), (x + w+10, y + h+10), (0, 255, 0), 3)
		crop = frame[y-10:y+h+10, x-10:x+w+10]
		time.sleep(0.1)
		#cv2.imshow("Crop", crop)
		cv2.imwrite('snapshot.jpg',crop)
		break
	# show the output frame
	cv2.imshow("Frame - Press q to quit", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
