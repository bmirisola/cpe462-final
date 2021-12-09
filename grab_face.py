import os
import cv2

webcam = cv2.VideoCapture(0)
path_img = os.path.abspath("saved_img.jpg")

while True:
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            try:
                os.mkdir("data")
                os.mkdir("data/person")
            except FileExistsError:
                pass
            cwd = os.getcwd()
            cv2.imwrite(filename="{}/{}/saved_img.jpg".format(cwd,"data/person"), img=frame)
            print("Processing image...")
            print("Image saved!")
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            print("quiting")
            break

webcam.release()
cv2.destroyAllWindows()
