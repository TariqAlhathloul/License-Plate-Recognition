import numpy as np
import cv2
from ultralytics import YOLO
import easyocr
import streamlit as st

# title of the app
st.title("Real-time License Plate Detection")

# load the model
model = YOLO('./best1.torchscript')

# function to get license plate text using EasyOCR
def get_license_plate(frame):
    reader = easyocr.Reader(['en', 'ar'])
    results = reader.readtext(frame, detail=0) 
    license_plate_text = " ".join(results)
    return license_plate_text

# start to capture frames from user webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Cannot open webcam")
    st.stop()

# get the webcam properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# display webcam properties in the app
st.write(f"Webcam resolution: {width}x{height} at {fps} FPS")

# Placeholder for the video frame
frame_placeholder = st.empty()


while cap.isOpened():
    success, frame = cap.read()

    if not success:
        st.error("Failed to capture frame from webcam.")
        break

    # send the frame to the model
    results = model(frame)

    # draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # send the frame to the OCR model
    license_plate_text = get_license_plate(annotated_frame)

    # display the license plate text on the frame if detected
    if license_plate_text:
        cv2.putText(annotated_frame, license_plate_text, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
