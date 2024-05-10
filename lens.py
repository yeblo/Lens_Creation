import argparse

import cv2
from imutils.video import VideoStream
from imutils import face_utils, translate, resize
import time
import dlib

import numpy as np


parser = argparse.ArgumentParser()
# parser.add_argument("-predictor", required=True, help="shape_predictor_68_face_landmarks.dat.bz2")
parser.add_argument("-predictor", required=True, help="path to predictor")
args = parser.parse_args()
print(args.predictor)
vs = VideoStream().start()
time.sleep(1.5)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.predictor)

eyelayer = np.zeros((600,800,3), dtype = 'uint8')
eye_mask = eyelayer.copy()
eye_mask = cv2.cvtColor(eye_mask, cv2.COLOR_BGR2GRAY)
translated = np.zeros((600,800,3), dtype = 'uint8')
translated_mask = eye_mask.copy()



class EyeList(object):
    def __init__(self, length):
        self.length = length
        self.eyes = []

    def push(self, newcoords):
        if len(self.eyes) < self.length:
            self.eyes.append(newcoords)
        else:
            self.eyes.pop(0)
            self.eyes.append(newcoords)
    
    def clear(self):
        self.eyes = []

eyelist = EyeList(10)
eyeSnake = False
while True:
    frame = vs.read()
    frame = resize(frame, width=800)
    # print(frame.shape)

    eyelayer.fill(0)
    eye_mask.fill(0)
    translated_mask.fill(0)
    translated.fill(0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray,0)
    if eyeSnake: 
        for rect in rects:
            #x,y,w,h = face_utils.rect_to_bb(rect)
            #cv2.rectangle(frame, (x,y),(x+w, y+h), (255,128,0),2)#coordinates, height, width and color 
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
    
            leftEye = shape[1:68]
            #rightEye = shape[42:48]

            cv2.fillPoly(eye_mask, [leftEye], 255)

            # cv2.fillPoly(eye_mask,[rightEye], 255)
            

            eyelayer = cv2.bitwise_and(frame, frame, mask=eye_mask)
            x,y,w,h = cv2.boundingRect(eye_mask)
            # cv2.rectangle(eyelayer, (x,y),(x+w, y+h), (255,128,0),2)

            eyelist.push([x,y])
            for i in reversed(eyelist.eyes):
                translated1 = translate(eyelayer, i[0] - x, i[1] - y)
                translated1_mask = translate(eye_mask, i[0] - x, i[1] - y)
                translated_mask = np.maximum(translated_mask, translated1_mask)
                translated = cv2.bitwise_and(translated, translated, mask=255 - translated1_mask)
                translated += translated1

            # for point in shape[36:48]:
            #     cv2.circle(frame, tuple(point), 2, (128,255,0))
        frame = cv2.bitwise_and(frame, frame, mask = 255 - translated_mask)
        frame += translated
    cv2.imshow("lens creation",frame)
    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break
    if key == ord("s"):
        eyeSnake = not eyeSnake
        eyelist.clear
cv2.destroyAllWindows()
vs.stop()