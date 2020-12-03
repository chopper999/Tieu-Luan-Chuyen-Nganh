
import numpy as np
import argparse
import imutils
import time
import cv2


class boundingbox_2():
	def run(self, frame, ct , net):

		self.frame = frame
		self.ct = ct
		self.net = net

		(H, W) = (None, None)

		if W is None or H is None:
			(H, W) = frame.shape[:2]


		blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
			(104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()
		

		rects = []

		for i in range(0, detections.shape[2]):

			if detections[0, 0, i, 2] > 0.5:

				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				rects.append(box.astype("int"))

				(startX, startY, endX, endY) = box.astype("int")
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 1)
		print(rects)
		objects = ct.update(rects)
		print(objects)


		for (objectID, centroid) in objects.items():

			text = "ID {}".format(objectID)


			cv2.putText(frame, text, (centroid[0] -10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		return frame
