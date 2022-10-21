# Streetview-Fibre-pole-detection-
This project was focused on utilizing deep learning algorithm to automatically detect fibre pole and its location in images gotten from google streetview  app. This approached was aimed ruducing the cost, time and labor involved in manually mapping utility poles for deploying 5g equipment
# Summary
Three object detection model were consider during the course of the project (yolov7, mask R cnn, EfficientDet)
Images collected from google streetview app and mobile phones were labelled and augmented using Roboflow and Makesense Ai to train the models.
Yolov7 model performed the best, it has a fast inference rate(5fps to 160fps) than other object detection model considered during this project
This solution was deployed to streamlit for easy access and utilitization

