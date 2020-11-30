import cv2
from PIL import Image
import os
import imutils
import numpy as np

from Detect_Face import detectface

class load_video():
	
	def	__init__(self, path):
		self.path = path
		#self.cam = cv2.VideoCapture(self.path)
		#__, self.frame = self.cam.read()
		#self.cam.release()

	def	show(self):
		arr = []
		
		cam = cv2.VideoCapture(self.path)
		de = detectface()


		while True:
			__, frame = cam.read()
			#cv2.imshow('frame',self.frame)
			#img = self.frame
			#img_item = "frame" + str(i) + ".png"
			#img = img.astype('float32')
			#img /= 255.0
			#arr.append(img)
			#re.runn(frame)
            #tr.run()

			de.run(frame)

			# cv2.imwrite( "imageFrame/"+ img_item, self.frame)
			#if cv2.waitKey(1) &0xFF == ord('q'):
			#	break
		
		cam.release()
		cv2.destroyAllWindows()


