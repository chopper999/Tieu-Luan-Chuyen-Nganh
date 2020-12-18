
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw
import imutils

import os


class draw_facenet():

    def draw_rectangle(self, frame, boxes):
        self.frame = frame
        self.boxes = boxes

        (startX, startY, endX, endY) = (None, None, None, None)
        
        try:
            for box in boxes:
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 1)
        except:
            pass
                
        return frame

    def draw_text(self, frame, boxes, names, sort_res, objects):
        self.frame = frame
        self.boxes = boxes
        self.names = names
        self.sort_res = sort_res
        self.objects = objects

        (startX, startY, endX, endY) = (None, None, None, None)
        i = 0
        try:
            for box in boxes:   
                (startX, startY, endX, endY) = box.astype("int")                             
                text = str(names[sort_res[i]])
                cv2.putText(frame, text, (startX + 10, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                i += 1
        except:
            pass

        try:
            for (objectPER, centroid) in objects.items():
                text = "Track {}".format(objectPER)
                cv2.putText(frame, text, (centroid[0] - 30, centroid[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.circle(frame, (centroid[0] , centroid[1]), 4, (255, 255, 0), -1)
        except:
            pass
        return frame


            

