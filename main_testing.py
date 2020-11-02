from Load_Camera import load_video
from Detect_Face import detectface




vi = load_video('videos/video22.mp4')
de = detectface()

#Module_1
#vi.show()

#Module_2
de.run()