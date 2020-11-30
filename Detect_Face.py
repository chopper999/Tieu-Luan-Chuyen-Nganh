import cv2
from mtcnn.mtcnn import MTCNN
import os, os.path
import imutils
from Recognizer_Face import recognizer
from Training_Face import training
from BoundingBox_Face import boundingbox



class detectface():
    def run(self, frame):
        self.frame = frame
        detector = MTCNN()
        re = recognizer()
        tr = training()
        bo = boundingbox()

        exit = True
        while exit:
            #frame = cv2.imread("imageFrame/" + "frame" + str(i) + ".png")
            #frame = image * 255.0
            #frame = frame.astype('unit-8')
            #print('Data Type: %s' % frame.dtype)
            ket_qua = detector.detect_faces(self.frame)
            #print(ket_qua)

            if ket_qua != []:
                for Nguoi in ket_qua:
                    face_box = Nguoi['box']

                    roi_ = self.frame[face_box[1]:face_box[1]+face_box[3], face_box[0]:face_box[0]+face_box[2]]
                    self.roi_Resize = imutils.resize(roi_, width=160, height= 160)
                    #img_label = "imgFace" + str(count) + ".png"
                    #cv2.imwrite( "imageFace/"+ img_label, roi_Resize)
                    
                    #re.runn(self.roi_Resize)
                    #tr.run()

                    bo.run(self.frame)
                    exit = False


                    #cv2.rectangle(frame,(face_box[0], face_box[1]),
                    #              (face_box[0]+face_box[2], face_box[1] + face_box[3]),
                    #              (0,255,0), 2)
            else:
                bo.run(self.frame)
                exit = False   
            #cv2.imshow('frame',frame)
            #count = count + 1
            #if cv2.waitKey(1) &0xFF == ord('q'):
            #    exit = False
            #    break
            