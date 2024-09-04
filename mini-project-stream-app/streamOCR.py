import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2
import pytesseract
# title of the app
st.title("Real-time License Plate Detection")

# load our model
model = YOLO('../License-Plate-Detection-and-Recognition/BestModel1/weights/best.pt')

def ocr(img):
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # fillter noise
    text = pytesseract.image_to_string(gray)
    return text
# start to capture frames from user webcam
cap = cv2.VideoCapture(0)  

# check if the frames are captured fine
if not cap.isOpened():
    st.error("Could not open webcam.")
else:
    # get the webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # write the properties to the app
    st.write(f"Webcam resolution: {width}x{height} at {fps} FPS")

    # placeholder for the frame
    frame_placeholder = st.empty()

    while True:
        success, frame = cap.read()

        # end streaming if there no frames captured
        if not success:
            st.error("Failed to capture frame from webcam.")
            break

        results = model(frame)

        # license plate text
        text = None

        # if there is a license plate detected
        if len(results[0].boxes.xyxy) > 0: 
            boxes = results[0].boxes.xyxy.tolist()

            # drow bboxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

            # crop the license plate from the image  
            x1, y1, x2, y2 = boxes[0]
            cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # ocr
            text = ocr(cropped_img)

        # write the text if detected
        if text:
            cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, use_column_width=True)


    # release the webcam
    cap.release()
    cv2.destroyAllWindows()