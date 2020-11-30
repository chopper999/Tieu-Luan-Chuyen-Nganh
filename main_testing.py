from Load_Camera import load_video
from Detect_Face import detectface
from Recognizer_Face import recognizer
from Training_Face import training
from BoundingBox_Face import boundingbox
import cv2



vi = load_video('videos/video333.mp4')
#de = detectface()
#re = recognizer('imageFace')
#tr = training('Data_Traning/embeddings.pickle')
#bo = boundingbox('Data_Traning/recognizer.pickle','Data_Traning/le.pickle')

#Module_1
vi.show()

#Module_2
#de.run()

#Module_3
#re.run()

#module_4
#tr.run()

#module_5
#bo.run()
