import os
import numpy as np
import cv2
import pytesseract
import streamlit as st
from requests import get
from PIL import Image
from ultralytics import YOLO

#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
#os.environ['TESSDATA_PREFIX'] = r'/opt/homebrew/bin/tesseract'

#os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# # test image
# image_path = r'C:\Users\DELL\OneDrive\Documents\week6-miniProject\تنزيل.png'
# mage = Image.open(image_path)
# extracted_text = pytesseract.image_to_string(mage)
# print(extracted_text)

# # Download Arabic trained data 
# def download(url, file_name):
#     tessdata_dir = 'C:/Users/DELL/AppData/Local/Programs/Tesseract-OCR/tessdata'
#     if not os.path.exists(tessdata_dir):
#         os.makedirs(tessdata_dir)  # Create tessdata directory if it does not exist
#     file_path = os.path.join(tessdata_dir, file_name)
#     with open(file_path, "wb") as file:
#         response = get(url)
#         file.write(response.content)

# download("https://github.com/tesseract-ocr/tessdata/raw/master/ara.traineddata", "ara.traineddata")



# Title of the app
st.title("Real-time License Plate Detection")

# model = YOLO(r'C:\Users\DELL\OneDrive\Documents\week6-miniProject\License-Plate-Recognition\mini-project-stream-app\best1.torchscript')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Cannot open webcam")
    st.stop()

# Get the webcam properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Display webcam properties in the app
st.write(f"Webcam resolution: {width}x{height} at {fps} FPS")

frame_placeholder = st.empty()

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        st.error("Failed to capture frame from webcam.")
        break

    # results = model(frame)
    # annotated_frame = results[0].plot()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform OCR on the gray frame
    arabic_text = pytesseract.image_to_string(gray_frame)
    print(f'******PlATE TEXT*******: {arabic_text}')

    cv2.putText(frame, arabic_text, (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    frame_placeholder.image(frame, use_column_width=True)


cap.release()
cv2.destroyAllWindows()