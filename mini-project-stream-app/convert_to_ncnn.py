

# to have better performance, on the mobilephone and raspberry pi
# i will convert the model to ncnn format 

# import YOLO from ultralytics
from ultralytics import YOLO

# load the model
model = YOLO('./weights/best.pt')

# export the model to ncnn format
model.export(format='ncnn')
