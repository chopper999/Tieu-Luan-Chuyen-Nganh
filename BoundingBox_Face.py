import cv2
from mtcnn.mtcnn import MTCNN
from Load_Camera import load_video
import os, os.path
import pickle
import numpy as np


class boundingbox():
    def __init__(self, path1, path2):
        self.path1 = path1
        self.path2 = path2

    def run(self):
        detector = MTCNN()

        DIR = 'imageFrame'
        sum_file = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

        print("[INFO] loading Openface...")
        embedder = cv2.dnn.readNetFromTorch("Library/openface_nn4.small2.v1.t7")
        recognizer = pickle.loads(open(self.path1, "rb").read())
        le = pickle.loads(open(self.path2, "rb").read())

        e = True
        count = 0
        while e:
            for i in range(30,sum_file):
                frame = cv2.imread("imageFrame/" + "frame" + str(i) + ".png")
                ket_qua = detector.detect_faces(frame)
                #print(ket_qua)

                if ket_qua != []:
                    for Nguoi in ket_qua:
                        face_box = Nguoi['box']
                        keypoints = Nguoi['keypoints']
                        Do_tin_cay = Nguoi['confidence']
                        roi_color = frame[face_box[1]:face_box[1]+face_box[3], face_box[0]:face_box[0]+face_box[2]]

                        faceBlob = cv2.dnn.blobFromImage(roi_color, 1.0 / 255, (96, 96),
                            (0, 0, 0), swapRB=True, crop=False)
                        embedder.setInput(faceBlob)
                        vec = embedder.forward()

                        preds = recognizer.predict_proba(vec)[0]
                        j = np.argmax(preds)
                        proba = preds[j]
                        name = le.classes_[j]

                        # draw the bounding box of the face along with the associated
                        # probability
                        text = "{}: {:.2f}%".format(name, Do_tin_cay * 100)
                        y = face_box[1] - 10 if face_box[1] - 10 > 10 else face_box[1] + 10
                        cv2.putText(frame, text, (face_box[0], y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


                        cv2.rectangle(frame,(face_box[0], face_box[1]),
                                              (face_box[0]+face_box[2], face_box[1] + face_box[3]),
                                              (0,255,0), 2)
                 
                        cv2.circle(frame,(keypoints['left_eye']), 2, (0,0,255), 2)
                        cv2.circle(frame,(keypoints['right_eye']), 2, (0,0,255), 2)
                        cv2.circle(frame,(keypoints['nose']), 2, (0,0,255), 2)
                        cv2.circle(frame,(keypoints['mouth_left']), 2, (0,0,255), 2)
                        cv2.circle(frame,(keypoints['mouth_right']), 2, (0,0,255), 2)
                cv2.imshow('frame',frame)
                count = count + 1

                if cv2.waitKey(1) &0xFF == ord('q'):
                    e = False
                    break