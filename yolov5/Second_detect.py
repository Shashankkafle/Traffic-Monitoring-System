from hashlib import new
from Object_Detection import *
import numpy as np
# from PIL import Image as im
import PIL
import pandas
# import tensorflow as tf
import torch
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, xyxy2xywh)


# VIDEO_DIR = 'D:/Traffic-Monitoring-System/Traffic-Survalance-with-Computer-Vision-and-Deep-Learning/VideoDataSets/1.mp4'

# cap = cv2.VideoCapture(VIDEO_DIR)
# ret, im = cap.read() 
# im =  PIL. Image. open("C:/Users/Acer/Desktop/cars.jpg")
# im = np. array(im)
model = DetectMultiBackend('D:/Traffic-Monitoring-System/yolov5/yolov5s.pt', device='cpu', dnn=False)


def detect(im): 
 im0=im.copy()
#  print('IM0',im0,'\n')
 imgsz=(640, 640)
 imgsz = check_img_size(imgsz, s=model.stride)
 im = cv2.resize(im,imgsz)
 img=im.copy()
 half = False
 model.warmup(imgsz=(1, 3, *imgsz), half=half)
 im = np.transpose(im,(2,0,1))
 
 
 im = torch.from_numpy(im).to('cpu')
 im = im.half() if half else im.float()  # uint8 to fp16/32
 im /= 255  # 0 - 255 to 0.0 - 1.0
 if len(im.shape) == 3:
     im = im[None]  # expand for batch dim
 
 pred = model(im, augment=False, visualize=False)
 pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
#  count=0
#  while(count<5):
#         x1,y1,x2,y2,a,b = pred[0][count].numpy().astype(int)
#         print('coords',x1,y1,x2,y2)
#         cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
#         count+=1

#  cv2.imshow('outpt from pred',img)
#  cv2.waitKey(0)

 
#  x1,y1,x2,y2,a,b = pred[0][0].numpy().astype(int)
#  print('coords',x1,y1,x2,y2)
#  cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
#  cv2.imshow('outpt from pred',img)
#  cv2.waitKey(0)

 
 tensorPoints = []
#  gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
 for i, det in enumerate(pred):
     for *xyxy, conf, cls in (det):
    #  for *xyxy, conf, cls in reversed(det):
        tensorPoints.append(xyxy)
        # print('tensorPoints',tensorPoints)

 boxes=[]    
 
 for coords in tensorPoints :
     point=[]
     for coord in coords:
      coord= coord.numpy()
      point.append(int(np.asscalar(coord))) 
     boxes.append(tuple(point)) 
 new_boxes=[]
 for box in boxes:
     newbox=[]
     newbox.append(box[1] if box[1]<box[3] else box[3] )
     newbox.append(box[1] if box[1]>box[3] else box[3])
     newbox.append(box[0] if box[0]<box[2] else box[2])
     newbox.append(box[0] if box[0]>box[2] else box[2])
     new_boxes.append(tuple(newbox))

#      new_boxes.append(tuple(newbox))
#  print('old_boxes',boxes,'\n\n')
#  print('New Boxes',new_boxes,'\n\n')  
#  for box in boxes:
#      cv2.rectangle(im0, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)
     
#  cv2.imshow('op from bxees',im0)
#  cv2.waitKey(0)    
 
 print('BOXES',boxes,'\n\n')
 print('new_BOXES',new_boxes,'\n\n')

 return(new_boxes)     
 



 
     



