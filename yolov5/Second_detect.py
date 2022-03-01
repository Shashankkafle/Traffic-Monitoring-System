from asyncio.windows_events import NULL
from hashlib import new
from Object_Detection import *
import numpy as np
import torch
from detectMultiBackend import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, xyxy2xywh)

    

    
class DETECTION(object):
    _defaults = {
        "model_image_size": (640, 640)
    }

    def __init__(self,path,device):
        self.model = DetectMultiBackend(path, device=device, dnn=False)
        self.__dict__.update(self._defaults) # set up default values
        self.device = device
        self.warmup()


    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def warmup(self):
        imgsz = self.model_image_size
        self.model.warmup(imgsz=(1, 3, *imgsz), half=False)

    def detect(self,im): 
        print(self.model_image_size)
        # imgsz = check_img_size(imgsz, s=model.stride)
        im = cv2.resize(im,self.model_image_size)
        half = False
        # model.warmup(imgsz=(1, 3, *imgsz), half=half)
        im = np.transpose(im,(2,0,1))
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

        tensorPoints = []
        for i, det in enumerate(pred):
            for *xyxy, _, _ in (det):
                tensorPoints.append(xyxy)

        boxes=[]    

        for coords in tensorPoints:
            point=[]
            for coord in coords:
                coord= coord.numpy()
                point.append(int(np.asscalar(coord))) 
            boxes.append(tuple(point)) 
        new_boxes=[]
        for box in boxes:
            newbox=[]
            newbox.append(box[1] if box[1]<box[3] else box[3])
            newbox.append(box[1] if box[1]>box[3] else box[3])
            newbox.append(box[0] if box[0]<box[2] else box[2])
            newbox.append(box[0] if box[0]>box[2] else box[2])
            new_boxes.append(tuple(newbox))


        return(new_boxes)     





    




