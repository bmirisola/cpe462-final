import os
import cv2

webcam = cv2.VideoCapture(0)

while True:
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            try:
                os.mkdir("data")
            except FileExistsError:
                pass
            cwd = os.getcwd()
            cv2.imwrite(filename="{}/{}/saved_img.jpg".format(cwd,"data"), img=frame)
            print("Processing image...")
            print("Image saved!")
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            print("quiting")
            break

webcam.release()
cv2.destroyAllWindows()
