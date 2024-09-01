import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# title of the app
st.title("Real-time License Plate Detection")

# load our model
model = YOLO('./best1.torchscript')


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

        # send the frame to the model
        results = model(frame)

        # draw bounding boxes on the frame
        annotated_frame = results[0].plot()

        # convert the frame to RGB
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        frame_placeholder.image(frame_rgb, use_column_width=True)

        # Stop the webcam feed if the user presses 'q'
        #if st.button("Stop Webcam", key='q'):
         #   break

    # release the webcam
    cap.release()
    cv2.destroyAllWindows()