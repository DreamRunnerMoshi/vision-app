import numpy as np
import argparse
import time
import cv2
import os

from detect import *

def main():

    vid = cv2.VideoCapture(0)

    labelsPath="yolo_v3/coco.names"
    cfgpath="yolo_v3/yolov3.cfg"
    wpath="yolo_v3/yolov3.weights"
    Lables=get_labels(labelsPath)
    CFG=get_config(cfgpath)
    Weights=get_weights(wpath)
    nets=load_model(CFG,Weights)
    Colors=get_colors(Lables)

    while(True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()
    
        # Display the resulting frame
        # cv2.imshow('frame', frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # load our input image and grab its spatial dimensions
        # image = cv2.imread("./yolo_v3/person.jpg")
        
        res=get_predection(frame,nets,Lables,Colors)
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # # show the output image
        cv2.imshow("frame", image)
        # cv2.waitKey()

if __name__== "__main__":
  main()