import cv2
import imutils
import random
import os
import numpy as np


# making an overall data dir and getting to the main img
os.mkdir("data")
cwd = os.getcwd()
key = cv2.waitKey(1)
image = cv2.imread("{}/saved_img.jpg".format(cwd))

# rotating the img
os.mkdir("data/rotate")


def rotate(image):
    for i in range(0, 4):
        Rotated_image = imutils.rotate(image, i * 45)
        cv2.imwrite("{}/data/rotate/R{}.jpeg".format(cwd, i), img=Rotated_image)
    if key == ord("s"):
        cv2.destroyAllWindows()


rotate(image)

# adding salt and pepper noise to the img
def add_noise(img):
    row, col = img.shape[:2]
    number_of_pixels = random.randint(100, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 255
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 0
    return img


os.mkdir("data/noise")
for j in range(0, 4):
    cv2.imwrite("{}/data/noise/sp{}.jpeg".format(cwd, j), img=add_noise(image))

image = cv2.imread("{}/saved_img.jpg".format(cwd))
# flip the img in 3 diff ways
os.mkdir("data/flip")
for f in range(-1, 2):
    fimage = cv2.flip(image, f)
    cv2.imwrite("{}/data/flip/f{}.jpeg".format(cwd, f), img=fimage)

# resizing the img
os.mkdir("data/rsize")
for s in range(1, 5):
    width = int(image.shape[1] * 0.25 * s)
    height = int(image.shape[0] * 0.25 * s)
    dsize = (width, height)
    simage = cv2.resize(image, dsize)
    cv2.imwrite("{}/data/rsize/rs{}.jpeg".format(cwd, s), img=simage)

os.mkdir("data/translate")
height, width = image.shape[:2]
for t in range(1, 5):
    theight, twidth = height / t, width / t
    T = np.float32([[1, 0, twidth], [0, 1, theight]])
    img_translation = cv2.warpAffine(image, T, (width, height))
    cv2.imwrite("{}/data/translate/T{}.jpeg".format(cwd, T), img=img_translation)
