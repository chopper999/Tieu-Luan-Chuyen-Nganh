import numpy as np
from Training_FaceNet import training_faceNet

tr = training_faceNet()
tran = tr.train_run()
print(tran)
'''labels = np.array(tran)[:, -1]
labels = labels.flatten() 
output = np.unique(labels, return_counts=True)
print(output)'''