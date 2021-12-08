import cv2
import imutils
import os

os.mkdir('data')
cwd = os.getcwd()
key = cv2. waitKey(1)
image = cv2.imread("{}/saved_img.jpg".format(cwd))
for i in range(0, 4):
    Rotated_image = imutils.rotate(image, i*45)
    cv2.imwrite("{}/data/f{}.jpeg".format(cwd, i), img=Rotated_image)

cv2.imshow("Rotated", Rotated_image)
cv2.waitKey(0)

if key == ord('s'):
    cv2.destroyAllWindows()
