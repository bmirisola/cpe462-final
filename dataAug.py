import cv2
import imutils
key = cv2. waitKey(1)
image = cv2.imread(r"saved_img.jpg")
for i in range(3):
    count = i
    Rotated1_image = imutils.rotate(image, angle=i*45)
    cv2.imwrite(filename='Rotate.jpg', img=Rotated1_image)

cv2.imshow("Rotated1", Rotated1_image)
cv2.waitKey(0)

if key == ord('s'):
    cv2.destroyAllWindows()
