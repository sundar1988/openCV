import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier ('haarcascade_eye.xml')
cam = cv2.VideoCapture(0)

while (True):
    tf, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ##face detection
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
##        roi_gray = gray(y:y+h, x:x+w)
##        roi_img = img(y:y+h, x:x+w)
        ##eye detection
        eyes = eye_cascade.detectMultiScale(gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img, (ex,ey), (ex+ew, ey+eh), (0,0,255), 2)
    cv2.imshow('Image', img)
    k = cv2.waitKey(50)
    if k == 27: #ESC key
        break
cam.release()
cv2.destroyAllWindows()
