import cv2
import os, os.path
import pickle
import numpy as np
import imutils

import csv

class boundingbox():

    def run(self, frame, detector, ct):

        self.frame = frame
        self.detector = detector
        self.ct = ct

        e = True
        rects = []
        Do_tin_cay = None

        keys, values = [], []

        while e:
            ket_qua = detector.detect_faces(self.frame)
            #print(ket_qua)        
            try:
                for Nguoi in ket_qua:
                    face_box = Nguoi['box']
                    keypoints = Nguoi['keypoints']
                    Do_tin_cay = Nguoi['confidence']
                    
                    cv2.rectangle(frame,(face_box[0], face_box[1]),
                                              (face_box[0]+face_box[2], face_box[1] + face_box[3]),
                                              (0,255,0), 2)
                    
                    cv2.circle(frame,(keypoints['left_eye']), 2, (0,0,255), 2)
                    cv2.circle(frame,(keypoints['right_eye']), 2, (0,0,255), 2)
                    cv2.circle(frame,(keypoints['nose']), 2, (0,0,255), 2)
                    cv2.circle(frame,(keypoints['mouth_left']), 2, (0,0,255), 2)
                    cv2.circle(frame,(keypoints['mouth_right']), 2, (0,0,255), 2)
                    rects.append(face_box)
                    print(rects)

                e = False
            except:
                e = False   
                   
        objects = ct.update(rects)
        
        for key, value in objects.items():
            keys.append(key)
            values.append(value)       

        with open("data.csv", "w") as outfile:
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(keys)
            csvwriter.writerow(values)


        for (objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0]*2 - 15, centroid[1]*2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.circle(frame, (centroid[0]*2, centroid[1]*2), 4, (255,255,0), -1)

        return frame
