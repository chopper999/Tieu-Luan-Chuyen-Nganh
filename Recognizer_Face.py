from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os


class recognizer():
	def	__init__(self, path):
		self.path = path

	def run(self):
		print("[INFO] loading face recognizer...")
		embedder = cv2.dnn.readNetFromTorch("Library/openface_nn4.small2.v1.t7")
		# modul dnn: Deep neural network - Deep Learning, dùng để kết nối tới các framwork - Torch
		#  nhóm dùng pretrain model để embedding các khuôn mặt có trong bức ảnh thành những vector embedding 128D
		print("[INFO] quantifying faces...")
		imagePaths = list(paths.list_images(self.path))

		knownEmbeddings = []
		knownNames = []

		total = 0
		for (i, imagePath) in enumerate(imagePaths):

			print("[INFO] processing image {}/{}".format(i + 1,
				len(imagePaths)))
			name = imagePath.split(os.path.sep)[-2]

			image = cv2.imread(imagePath)
			# tiền xử lý ảnh, giảm thiểu sự ảnh hưởng của sự thay đổi màu sắc do ánh sáng
			#  blob = cv2.dnn.blobFromImages(images, scalefactor=1.0, size, mean, swapRB=True)
			# size kích thước không gian
			# đầu ra là đốm màu - blob 4 chiều dùng cho neural network
			faceBlob = cv2.dnn.blobFromImage(image, 1.0//255.0,
						(96,96), (0, 0, 0), swapRB=True, crop=False)
			
			# cho các blob vào embedder để thu được các vector 128D mô tả khuôn mặt
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

		# Lưu trữ vào pickle
		print("serializing {} encodings...".format(total))
		data = {"embeddings": knownEmbeddings, "names": knownNames}
		print(knownNames)
		f = open("Data_Traning/embeddings.pickle", "wb")
		f.write(pickle.dumps(data))
		f.close()










		# Doc File Pickle
		"""pickle_file = open("Data_Traning/embeddings.pickle", "rb")
		objects = []
		while True:
		    try:
		        objects.append(pickle.load(pickle_file))
		    except EOFError:
		        break
		pickle_file.close()
		print(objects)"""

