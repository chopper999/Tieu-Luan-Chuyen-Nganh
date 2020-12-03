
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display
import imutils

import csv

class boundingbox_facenet():
    def run(self, frame, facenet, ct):
        self.frame = frame
        self.facenet = facenet
        self.ct = ct

        keys, values = [], []

        (startX, startY, endX, endY) = (None, None, None, None)
        #frame = imutils.resize(frame, width=1600)
        frames = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        rects = []
        e = True
        while e:
            # Detect faces
            boxes, _ = facenet.detect(frames)
            #print(boxes)
            try:
                for box in boxes:
                    rects.append(np.array(box.astype("int")))
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 1)
                e = False
            except:
                e = False
                #pass
        objects = ct.update(np.array(rects))

        
        for key, value in objects.items():
            keys.append(key)
            values.append(value)       

        with open("data.csv", "w") as outfile:
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(keys)
            csvwriter.writerow(values)

        try:
            for (objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.circle(frame, (centroid[0] , centroid[1]), 4, (255, 255, 0), -1)
        except:
            pass
        return frame
            


