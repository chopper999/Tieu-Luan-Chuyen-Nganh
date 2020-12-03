import cv2
#from PIL import Image
#import os
import imutils
import numpy as np
#from mtcnn.mtcnn import MTCNN


from facenet_pytorch import MTCNN
import torch


from Tracking.centroidtracker import CentroidTracker
from BoundingBox_Face import boundingbox
from BoundingBox_Face_2 import boundingbox_2
from BoundingBox_Face_Facenet import boundingbox_facenet


cam = cv2.VideoCapture("videos/video333.mp4")
ct = CentroidTracker()
bo = boundingbox()
bo2 = boundingbox_2()
bo3 = boundingbox_facenet()


print("loading model : DNN and MTCNN")
net = cv2.dnn.readNetFromCaffe("Library/deploy.prototxt", "Library/res10_300x300_ssd_iter_140000.caffemodel")
embedder = cv2.dnn.readNetFromTorch("Library/openface_nn4.small2.v1.t7")

#detector = MTCNN()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

facenet = MTCNN(keep_all=True, device=device)

while True:
	__, frame = cam.read()

	#frame = bo.run(frame, detector, ct)
	#frame = bo2.run(frame, ct, net)
	frame = bo3.run(frame, facenet,ct)


	cv2.imshow('frame',frame)
	if cv2.waitKey(1) &0xFF == ord('q'):
		break
	
cv2.destroyAllWindows()
cam.stop()
