
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw
import imutils


import os


class draw_facenet():

    def draw_rectangle(self, frame, boxes, names, sort_res):
        self.frame = frame
        self.boxes = boxes
        self.names = names
        self.sort_res = sort_res

        (startX, startY, endX, endY) = (None, None, None, None)
        try:
            for box in boxes:
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 1)
        except:
            pass
                
        return frame

    def draw_tracking(self, frame, boxes, objects):
        self.frame = frame
        self.boxes = boxes
        self.objects = objects

        (startX, startY, endX, endY) = (None, None, None, None)
        try:
            for (objectPER, centroid) in objects.items():
                text = "Track {}".format(objectPER)
                cv2.putText(frame, text, (centroid[0] - 30, centroid[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.circle(frame, (centroid[0] , centroid[1]), 4, (255, 255, 0), -1)

        except:
            pass
        return frame

    def draw_text_train(self, frame, boxes, names, sort_res):
        self.frame = frame
        self.boxes = boxes
        self.names = names
        self.sort_res = sort_res

        (startX, startY, endX, endY) = (None, None, None, None)
        print(sort_res)
        try:
            for box in boxes:   
                (startX, startY, endX, endY) = box.astype("int")                             
                text = str(names[sort_res[i]])
                x = startX + 10
                y = startY -10
                cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        except:
            pass
        return frame





            


