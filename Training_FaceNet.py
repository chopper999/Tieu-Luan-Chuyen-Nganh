import numpy as np
import os

from sklearn.neighbors import KNeighborsClassifier 


class training_faceNet():
	def train_run(self):		
		face_dataset = []
		labels = []
		class_id = 0
		names = {}
		dataset_path = dataset_path = "./facenet_model/"
		# Dataset
		for fx in os.listdir(dataset_path):
		    if fx.endswith('.npy'):
		        names[class_id] = fx[:-4]
		        data_item = np.load(dataset_path + fx, allow_pickle=True)
		        face_dataset.append(data_item)

		        target = class_id * np.ones((data_item.shape[0],))
		        class_id += 1
		        labels.append(target)

		face_dataset = np.concatenate(face_dataset, axis=0)
		face_labels = np.concatenate(labels, axis=0).reshape((-1, 1)) 
		#print(face_dataset.shape)
		#print(face_labels.shape)  

		trainset = np.concatenate((face_dataset,face_labels ), axis=1)
		#trainset = np.concatenate(trainset, axis=0)
		return trainset
