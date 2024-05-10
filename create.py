import argparse

import cv2
from imutils.video import VideoStream
from imutils import face_utils, translate, resize
import time
import dlib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-predictor", required=True, help="path to predictor")
args = parser.parse_args()
print(args.predictor)
vs = VideoStream().start()
time.sleep(1.5)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.predictor)
while True:
    frame = vs.read()
    frame = resize(frame, width = 800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        x,y,w,h = face_utils.rect_to_bb(rect)
        #cv2.rectangle(frame, (x,y),(x+w, y+h),(127,255,212),2)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for point in shape:
            cv2.circle(frame, tuple(point), 2, (128,255,0))

    cv2.imshow("Create Lens", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()