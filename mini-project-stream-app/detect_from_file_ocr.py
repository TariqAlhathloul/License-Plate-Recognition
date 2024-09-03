import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import easyocr
import io

# load model
model = YOLO('./best1.torchscript')

# title of the app
st.title("License Plate Detection")
st.write("Upload an image to detect license plate.")

# function to convert 
def convet_to_opencv_image(image):
    image = Image.open(image)
    image_np = np.array(image)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2

# get the license plate text using easyocr
def get_license_plate(frame):
    reader = easyocr.Reader(['en'])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    noise_flt = cv2.bilateralFilter(gray, 11, 17, 17)
    results = reader.readtext(noise_flt) 
    license_plate_text = results[0][-2]
    return license_plate_text

# convert the image to PIL image then to bytes
def convert_to_downloadable(image):
    pil_image = Image.fromarray(image)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# file uploader
image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image is not None:

    # convert to open cv image
    image = convet_to_opencv_image(image)
    
    # send the image to the model
    results = model(image)
    # draw bbox on the image
    annotated_image = results[0].plot()

    # convert back to RGB
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # send the frame to the OCR model
    text = get_license_plate(image)

    if text:
        #cv2.putText(annotated_image, text, (results[0], results[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        st.write(f"Detected License Plate Number: {text}")

    # display the image
    st.image(annotated_image, caption='Detected License Plate', use_column_width=True)

    # download the image
    annotated_image = convert_to_downloadable(annotated_image)
    #st.download_button('Download Annotated Image', annotated_image)
    st.download_button(label='Download Annotated Image',
                       data=annotated_image,
                       file_name='annotated_image.png',
                       mime='image/png')