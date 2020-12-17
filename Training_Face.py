
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

class training():

	def run(self):
		print("[INFO] loading face embeddings...")
		data = pickle.loads(open("Data_Traning1/embeddings.pickle", "rb").read())

		print("[INFO] encoding labels...")
		le = LabelEncoder()
		labels = le.fit_transform(data["names"])

		print("[INFO] training model...")
		recognizer = SVC(C=1.0, kernel="linear", probability=True)
		recognizer.fit(data["embeddings"], labels)



		#Save training
		f = open("Data_Traning1/recognizer.pickle", "wb")
		f.write(pickle.dumps(recognizer))
		f.close()

		f = open("Data_Traning1/le.pickle", "wb")
		f.write(pickle.dumps(le))
		f.close()

		# Doc File Pickle
		pickle_file = open("Data_Traning1/recognizer.pickle", "rb")
		objects = []
		while True:
		    try:
		        objects.append(pickle.load(pickle_file))
		    except EOFError:
		        break
		pickle_file.close()
		#print(objects)