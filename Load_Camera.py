import cv2
#from PIL import Image
#import os
#import imutils
import numpy as np

from Detect_Face import detectface
from BoundingBox_Face import boundingbox

class load_video():
	
	def	__init__(self, path, rects):
		self.path = path
		self.rects = rects

	def	show(self):
		cam = cv2.VideoCapture(self.path)
		#de = detectface()
		bo = boundingbox()


		while True:
			__, frame = cam.read()
			#de.run(frame)
			bo.run(frame, self.rects)
	

