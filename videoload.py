import cv2
import numpy as np

cam = cv2.VideoCapture(0)
#print cam.get(3)

while (True):                               #info from cam.read
    tf, frame = cam.read()
    #(true/false, frame, dtype=uint8)->0-255
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)# change the color
    cv2.imshow('Single Frame', frame)
    key = cv2.waitKey(1) # frame rate 1 millisecond
    if key ==27: #esc key
        break
cam.release()

cv2.destroyAllWindows()
