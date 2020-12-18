import numpy as np

import torch

class recognizer_faceNet():
	def process_faces(self,faces,resnet):
		self.faces = faces
		self.resnet = resnet
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		#faces = [f for f in faces if f is not None]

		faces = torch.cat(tuple(faces)).to(device)

		embeddings = resnet(faces.unsqueeze(0))

		t2 = torch.tensor(embeddings.data, requires_grad=False)

		#centroid = embeddings.mean(dim=0)
		#x = (embeddings - centroid).norm(dim=0).cpu()
		t2 = t2.to("cpu")
		t2 = np.array(t2)
		t2 = t2.reshape((t2.shape[0], -1))

		return t2


		

'''class dis():
	def distance(self, v1, v2):
		x = np.sqrt(((v1-v2)**2).sum())
		return x

class classification_knn():

	def knn(self, train, test, k=5):
		dx = dis()
		dist = []
		
		for i in range(train.shape[0]):
			# Get the vector and label
			ix = train[i, :-1]
			iy = train[i, -1]
			# Compute the distance from test point
			d = dx.distance(test, ix)
			dist.append([d, iy])
		# Sort based on distance and get top k
		dk = sorted(dist, key=lambda x: x[0])[:k]
		# Retrieve only the labels
		labels = np.array(dk)[:, -1]
		
		# Get frequencies of each label
		output = np.unique(labels, return_counts=True)
		# Find max frequency and corresponding label
		index = np.argmax(output[1])
		return output[0][index]'''

