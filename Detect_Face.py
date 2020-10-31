import cv2
from mtcnn.mtcnn import MTCNN
import os, os.path
import imutils



class detectface():
    def run(self):
        detector = MTCNN()
        DIR = 'imageFrame'
        sum_frame = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        count = 0

        exit = True
        while exit:
            for i in range(30,sum_frame):
                frame = cv2.imread("imageFrame/" + "frame" + str(i) + ".png")
                ket_qua = detector.detect_faces(frame)
                #print(ket_qua)

                if ket_qua != []:
                    for Nguoi in ket_qua:
                        face_box = Nguoi['box']



                        roi_ = frame[face_box[1]:face_box[1]+face_box[3], face_box[0]:face_box[0]+face_box[2]]
                        roi_Resize = imutils.resize(roi_, width=160, height= 160)
                        img_label = "imgFace" + str(count) + ".png"
                        cv2.imwrite( "imageFace/"+ img_label, roi_Resize)

                        cv2.rectangle(frame,(face_box[0], face_box[1]),
                                      (face_box[0]+face_box[2], face_box[1] + face_box[3]),
                                      (0,255,0), 2)
                    
                cv2.imshow('frame',frame)
                count = count + 1
                if cv2.waitKey(1) &0xFF == ord('q'):
                    exit = False
                    break
            