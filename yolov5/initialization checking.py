from Second_detect import DETECTION
import cv2
import numpy as np


def letterbox_image(image, size):
    '''resize image with hangedunc aspect ratio using padding'''
    ih, iw = image.shape[:2]
    h, w = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw,nh), cv2.INTER_CUBIC)
    new_image = np.zeros((w, h, 3), np.uint8)
    new_image[:,:] = (128, 128, 128)
    new_image[(h-nh)//2:(h-nh)//2 +nh, (w-nw)//2:(w-nw)//2 + nw] = image
    return new_image

detection = DETECTION('D:/Traffic-Monitoring-System/yolov5/yolov5s.pt','cpu')

image = cv2.imread('C:/Users/Acer/Desktop/ACE074BCT070.jpg')
image = letterbox_image(image, (640,640))
print(detection.detect(image))