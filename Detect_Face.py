import cv2
from mtcnn.mtcnn import MTCNN
from BoundingBox_Face import boundingbox


class detectface():
    def run(self, frame):
        self.frame = frame
        bo = boundingbox()

        exit = True
        while exit:
                          
            bo.run(self.frame)
            exit = False
            