import streamlit as st
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from PIL import Image


def convet_to_opencv_image(image):
    image = Image.open(image)
    image_np = np.array(image)
    image_cv = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
    return image_cv
model = YOLO('./best1.torchscript')


st.title("License Plate Detection")
st.write("Upload an image to detect license plate.")

# file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # convert to open cv image
    image = convet_to_opencv_image(uploaded_file)
    
    # send the image to the model
    results = model(image)

    # draw bbox on the image
    annotated_image = results[0].plot()

    # convert back to RGB
    annotated_image = cv.cvtColor(annotated_image, cv.COLOR_BGR2RGB)
    
    # Display the image
    st.image(annotated_image, caption='Detected License Plate', use_column_width=True)
