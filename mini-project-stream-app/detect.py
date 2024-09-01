import numpy as np
import cv2 as cv
from ultralytics import YOLO
# load the model

model = YOLO('./best1.torchscript')

# start to capture frames from user webcam

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("cannot open webcam")
    exit()

while True:

    success, frame = cap.read()

    if not success:
        print("cannot read frame")
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    cv.imshow('frame', annotated_frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
