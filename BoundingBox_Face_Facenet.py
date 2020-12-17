
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw
#from IPython import display
import imutils

import os


class boundingbox_facenet():

    def run(self, frame, facenet, resnet, detector, ct, re, ClassKNN, tr):
        self.frame = frame
        self.facenet = facenet
        self.resnet = resnet
        self.detector = detector
        self.ct = ct
        self.re = re
        self.ClassKNN = ClassKNN
        self.tr = tr


        (startX, startY, endX, endY) = (None, None, None, None)
        #frame = imutils.resize(frame, width=1600)
        frames = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        rects = []
        img_crop = []
        faces = []
        X = []
        dataset_path = "./facenet_model/"
        class_id = 0
        names = {}
        textmodel = []
        # Detect faces
        boxes, _ = facenet.detect(frames)
        
        try:
            for box in boxes:

                rects.append(np.array(box.astype("int")))
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 1)

            for box2 in boxes:

                (startX, startY, endX, endY) = box2.astype("int")
                roi = frame[startY :endY, startX: endX]
                roi_Resize = imutils.resize(roi, width=160, height= 160)
                img_crop.append(roi_Resize)

                faces.extend(self.detector(img_crop))

        except:
            pass
            
        objects = ct.update(np.array(rects))

        for f in faces:
           if f is not None:           
                X.append(re.process_faces(f,resnet))
                X = np.array(X)
                X = X.transpose(2,0,1).reshape(3,-1)

        for (name, xxx) in objects.items():
            if X != []:

                if not os.path.exists(dataset_path + "person" + str(name) + '.npy'):
                    with open(dataset_path + "person" + str(name)+ '.npy', 'wb') as f:
                        np.save(f , X)
                        print ("Dataset saved at : {}".format(dataset_path + "person" + str(name) + '.npy'))
                else:
                    with open(dataset_path + "person" + str(name) + '.npy', 'rb') as f:
                        out = np.load(f,  allow_pickle=True)
                        output = np.concatenate((out,X))
                      
                        #Loc du lieu trung lap

                        #out = array_in.reshape((array_in.shape[0], -1)) 
                        #print(output)
                        
                    with open(dataset_path + "person" + str(name)+ '.npy', 'wb') as f:
                        np.save(f , output)


        for fx in os.listdir(dataset_path):
            if fx.endswith('.npy'):
                names[class_id] = fx[:-4]

        train_dataset = tr.train_run()
        #print(train_dataset)
        
        for face_sec in X:
            newtext = ClassKNN.knn(train_dataset, face_sec.flatten())
            print(int(newtext))

            try:
                for (objectPER, centroid) in objects.items():
                    #text = "Person {}".format(objectPER)
                    text = int(newtext)

                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.circle(frame, (centroid[0] , centroid[1]), 4, (255, 255, 0), -1)

            except:
                pass
    
        return frame
            


