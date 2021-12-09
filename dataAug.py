import cv2
import imutils
import random
import os
import numpy as np


# making an overall data dir and getting to the main img
# When you read this please remember to always pull before you start working on a project.
# It's a big time saver trust me (;

key = cv2.waitKey(1)
path = os.path.abspath("data/person")
image = cv2.imread("{}/saved_img.jpg".format(path))
# rotating the img
def rotate(image):
    for i in range(0, 4):
        Rotated_image = imutils.rotate(image, i * 45)
        cv2.imwrite("{}/R{}.jpeg".format(path, i), img=Rotated_image)
    if key == ord("s"):
        cv2.destroyAllWindows()


rotate(image)

# adding salt and pepper noise to the img
def add_noise(img):
    row, col = img.shape[:2]
    number_of_pixels = random.randint(100, 1000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 255
    number_of_pixels = random.randint(300, 1000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 0
    return img


for j in range(0, 4):
    cv2.imwrite("{}/sp{}.jpeg".format(path, j), img=add_noise(image))

# flip the img in 3 diff ways
for f in range(-1, 2):
    fimage = cv2.flip(image, f)
    cv2.imwrite("{}/f{}.jpeg".format(path, f), img=fimage)

# resizing the img
for s in range(1, 5):
    width = int(image.shape[1] * 0.25 * s)
    height = int(image.shape[0] * 0.25 * s)
    dsize = (width, height)
    simage = cv2.resize(image, dsize)
    cv2.imwrite("{}/rs{}.jpeg".format(path, s), img=simage)

# translating the img
height, width = image.shape[:2]
for t in range(1, 5):
    theight, twidth = height / t, width / t
    T = np.float32([[1, 0, twidth], [0, 1, theight]])
    img_translation = cv2.warpAffine(image, T, (width, height))
    cv2.imwrite("{}/T{}.jpeg".format(path, T), img=img_translation)

# Cropping an image
for c in range(1, 5):
    cropped_image = image[100 * c : 200 * c, 20 * c : 300 * c]
    cv2.imwrite("{}/c{}.jpg".format(path, c), cropped_image)
