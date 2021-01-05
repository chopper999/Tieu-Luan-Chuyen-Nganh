
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw
import imutils
import os

#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#SVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import itertools
from draw_face import draw_facenet


class Process():
    def run(self, frame, facenet, resnet, ct, re, tr, vitrimodel):
        self.frame = frame
        self.facenet = facenet
        self.resnet = resnet
        self.ct = ct
        self.re = re
        self.tr = tr
        self.vitrimodel = vitrimodel

        draw = draw_facenet()

        (startX, startY, endX, endY) = (None, None, None, None)
        frames = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        rects = []
        img_crop = []
        faces = []
        X = []
        dataset_path = vitrimodel
        class_id = 0
        names = {}
        empty = False

        # Detect faces in frame
        boxes, _ = facenet.detect(frames)
        
        try:
            for box2 in boxes:
                rects.append(np.array(box2.astype("int")))

            for box in boxes:            
                (startX, startY, endX, endY) = box.astype("int")
                img_face = frame[startY :endY, startX: endX]
                roi_Resize = imutils.resize(img_face, width=160, height= 160)
                img_crop.append(roi_Resize)

                #detect embedding in faces
                faces.extend(facenet(img_crop))
   
        except:
            pass
            
        objects = ct.update(np.array(rects))

        for f in faces:
            if f is not None:
                X = list(X)
                X.append((re.process_faces(f,resnet)))
                X = np.array(X)
                #X = X.reshape(3,-1)
                X = X.transpose(2,0,1).reshape(3,-1)
            else:
                break


        for (name, xxx) in objects.items():
            textfile = dataset_path + "person" + str(name) + '.npy'
            if X != []:

                if not os.path.exists(textfile):
                    with open(textfile, 'wb') as f:
                        np.save(f , X)
                        print ("Dataset saved at : {}".format(textfile))
                else:
                    with open(textfile, 'rb') as f:
                        out = np.load(f,  allow_pickle=True)
                        output = np.concatenate((out,X))

                        #Loc du lieu trung lap
                        output_2 = np.unique(output, axis=0)
                        #out = array_in.reshape((array_in.shape[0], -1)) 
                        
                    with open(textfile, 'wb') as f:
                        np.save(f , output_2)


        for fx in os.listdir(dataset_path):
            if fx.endswith('.npy'):
                names[class_id] = fx[:-4]
                empty = True
                class_id += 1

        sort_res = None
        if empty == True and class_id > 2:

            train_dataset = tr.train_run()
            xx, yy = np.array(train_dataset)[:, 0:-1], np.array(train_dataset)[:, -1]

            #X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2)

            #model = KNeighborsClassifier(n_neighbors = 3)
            #model.fit(xx,yy)
            parameter_candidates = [
                {'C': [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000], 'kernel': ['linear']},
            ]

            clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)

            #clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            clf.fit(xx, yy)

            if X != []:
                #print(X)
                #response = model.predict(X)
                response = clf.predict(X)
                print(response)

        
                # Xu ly du doan
                sort_res = np.sort(response)
                sort_res = [[x, len(list(y))] for x, y in itertools.groupby(sort_res)]
                sort_res = sorted(sort_res,key=lambda x: x[1])
                sort_res.reverse()
                sort_res = np.array(sort_res)
                sort_res = sort_res[:,0]


                #print("muc do du doan :",metrics.accuracy_score(y_test, response))

        frame = draw.draw_rectangle(frame, boxes)
        frame = draw.draw_text_train(frame, boxes , names, sort_res)
        frame = draw.draw_tracking(frame, boxes, objects)

        return frame
            


