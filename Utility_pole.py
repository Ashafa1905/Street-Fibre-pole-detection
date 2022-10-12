import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import requests
import random
import onnxruntime as ort
from pathlib import Path
from collections import OrderedDict,namedtuple
#import tensorflow as tf
#import tensorflow_hub as hub
import time, sys
from streamlit_embedcode import github_gist
import urllib.request
import urllib
import matplotlib.pyplot as plt

@st.cache
def load_image(img):
    im = Image.open(img)
    return im


cuda = False
w = "C:/Users/CHIBUIKEM EFUGHA/OneDrive/Documents/streamlit object detection/pole_model.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)


def letterbox(
    im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, r, (dw, dh)

def Detection(our_image):
    # Name of the classes according to class indices.
    # Make sure all your classes are in this list in the right order!!
    
    #img = cv2.imread("C:/Users/CHIBUIKEM EFUGHA/OneDrive/Documents/streamlit object detection/tempDir/image_file")
    names = ["fibre-pole"]

    # Creating random colors for bounding box visualization.
    colors = {
        name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)
    }
    img = cv2.imread(our_image)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocessing the image for prediction.
    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    im.shape

    # Getting onnx graph input and output names.
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: im}

    # Running inference using session.
    outputs = session.run(outname, inp)[0]


    ori_images = [img.copy()]

    # Visualizing bounding box prediction.
    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0, y0, x1, y1])
        box -= np.array(dwdh * 2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 3)
        name = names[cls_id]
        color = colors[name]
        name += " " + str(score)
        cv2.rectangle(image, box[:2], box[2:], color, 2)
        cv2.putText(
            image,
            name,
            (box[0], box[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            [225, 255, 255],
            thickness=2,
        )

    return Image.fromarray(ori_images[0])



def load_image(image_file):
        img = Image.open(image_file)
        return img
def save_uploadedfile(uploadedfile):
     with open(os.path.join("tempDir", uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

def main():
    """Fibre Utility Pole Detection App"""
    st.title("Fibre Utility Pole Detection")
    new_title = '<p style="font-size: 42px;">Welcome to our Object Detection App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown("""
    This project was built using Streamlit and OpenCV 
    to demonstrate YOLO Object detection in both Street addresses
    and images."""
    )
    activities = ["Utility Pole Detection(Image)", "Utility Pole Detection(Street Address)", "About Us"]
    choice =  st.sidebar.selectbox("Select Activity", activities)
    if choice == "Utility Pole Detection(Image)":
        st.subheader("Image")
        image_file = st.file_uploader("Upload Image", type = ["jpg", "png", "jpeg"])

        if image_file is not None:
            file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
            st.text("Original Image")
            st.image(load_image(image_file), width=300)
             #Saving upload
            with open(os.path.join("tempDir", image_file.name),"wb") as f:
                f.write((image_file).getbuffer())
            st.success("Image Saved")
        
        # enhance_type = st.sidebar.radio("Enhance_type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
        # if enhance_type == "Gray-Scale":
        #     new_img = np.array(image_file.convert("RGB"))
        #     img = cv2.cvtColor(new_img, 1)
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     #st.write(new_img)
        #     st.image(gray)
        # if enhance_type == "Contrast":
        #     c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
        #     enhancer = ImageEnhance.Contrast(image_file)
        #     img_output = enhancer.enhance(c_rate)
        #     st.image(img_output)
        # if enhance_type == "Brightness":
        #     c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
        #     enhancer = ImageEnhance.Brightness(image_file)
        #     img_output = enhancer.enhance(c_rate)
        #     st.image(img_output)
        # if enhance_type == "Blurring":
        #     new_img = np.array(image_file.convert("RGB"))
        #     blur_rate = st.sidebar.slider("Blurring", 0.5, 3.5)
        #     img = cv2.cvtColor(new_img, 1)
        #     blur_img = cv2.GaussianBlur(img, (11,11), blur_rate)
        #     st.image(blur_img)
        # else:
        #     st.image(image_file, width=300)
        
    # 
    # #Utility Pole detection
    #     task = ["Object Detection(Image)", "Object Detection (Google street link)"]
    #     new_choice = st.sidebar.selectbox("Choose task", task)
        if st.button("Process"):
            img_placeholder = st.empty()
            # if new_choice == "Object Detection(Image)":
                
            try:
                image_path = 'tempDir\{}'.format(file_details['filename'])
                st.text('Detecting fibre Pole, please wait.....:)')
                # img1 = cv2.imread("image_path")
                # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                predicted_image = Detection(image_path)
                st.success('Detection COMPLETE, BYE')
                # Display the image with the detections in the Streamlit app
                img_placeholder.image(predicted_image, channels='BGR', width=300)
                st.success("Found {} number of poles".format(predicted_image))

            except KeyboardInterrupt:
                pass
                #st.image(result_img)
                #st.success("Found {} Poles".format(len(result_poles)))

    if choice ==   "Utility Pole Detection(Street Address)":
        st.subheader("Street Address")
        pass 


    elif choice == "About Us":
        st.subheader("About")

        

if __name__ == '__main__':
                main()	