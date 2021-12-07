import cv2
import imutils
#import os


key = cv2. waitKey(1)
image = cv2.imread("/Users/riyashrestha/cpe462-final/saved_img.jpg")
for i in range(0, 4):
    Rotated_image = imutils.rotate(image, i*45)
    cv2.imwrite("f{}.jpeg".format(i), img=Rotated_image)

cv2.imshow("Rotated", Rotated_image)
cv2.waitKey(0)

if key == ord('s'):
    cv2.destroyAllWindows()
