import cv2
import imutils
import numpy as np

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

from Tracking.centroidtracker import CentroidTracker
from Detect_Facenet import detect_facenet
from Recognizer_FaceNet import recognizer_faceNet
from Training_FaceNet import training_faceNet


cam = cv2.VideoCapture("videos/video333.mp4")
ct = CentroidTracker()

bo3 = detect_facenet()
re = recognizer_faceNet()

tr = training_faceNet()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

facenet = MTCNN(keep_all=True, device=device)

detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()
resnet = InceptionResnetV1(pretrained='vggface2',device=device).eval()
resnet.classify = True

while True:
	__, frame = cam.read()

	frame = bo3.run(frame, facenet, resnet, detector , ct, re , tr)

	cv2.imshow('frame',frame)
	if cv2.waitKey(1) &0xFF == ord('q'):
		break

cv2.destroyAllWindows()
cam.stop()
