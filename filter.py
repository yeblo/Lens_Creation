import cv2
import numpy as np 
import dlib
from math import hypot
cap = cv2.VideoCapture(0)
cap_image = cv2.imread("animated-mouth-png-1.png")


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        left_eyebrow = (landmarks.part(19).x, landmarks.part(19).y)
        right_eyebrow = (landmarks.part(24).x, landmarks.part(24).y)

        brow_width = int(hypot(left_eyebrow[0] - right_eyebrow[0], left_eyebrow[1] - right_eyebrow[1]) )
        brow_height = int(brow_width * 0.0012)

        brow_center = int((brow_width + brow_height)/2)


        top_left = ((brow_center - brow_width), (brow_center - brow_height))
        bottom_right = ((brow_center + brow_width), (brow_center + brow_height))

        grad_cap = cv2.resize(cap_image,(brow_width , brow_height))
        

        # print (brow_width)

    cv2.imshow("Filter creation", frame)  
    cv2.imshow("Cap", grad_cap)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
