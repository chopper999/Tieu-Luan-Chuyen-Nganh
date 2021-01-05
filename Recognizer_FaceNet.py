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


		

