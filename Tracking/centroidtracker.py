from scipy.spatial import distance as dist		#Dùng để tính khoảng cách
from collections import OrderedDict				#Sử dụng giống như mảng, nhưng ghi nhớ thứ tự các đối tượng truyền vào
import numpy as np

class CentroidTracker():
	def __init__(self, maxDisappeared=50):
		# maxDisappeared: số lượng khung hình liên tiếp tối đa mà một đối tượng phải mất đi khi không được track nữa
		self.nextObjectID = 0		#ID mới, chưa được dùng để đăng ký
		self.objects = OrderedDict()	#chưa ID và trọng tâm của đối tượng đó
		self.disappeared = OrderedDict()	#số lượng khung hình liên tiếp của 1 đối tượng bị đánh dấu là mất

		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# đăng ký theo dõi đối tượng với ID mới
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1 # tăng giá trị để gán cho ID tiếp theo

	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects): #Tham so truyen vao la cac bounding box
		#Neu khong co bounding box nao trong list
		if len(rects) == 0:
			# duyệt qua danh sách các đối tượng đã theo dõi và tăng giá trị disappeared lên
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# khi giá trị disappeared dat den maxDisappeared thì hủy đăng ký
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			return self.objects
 
		inputCentroids = np.zeros((len(rects), 2), dtype="int")	#khởi tạo mảng 0 để lưu trữ các giá trị trọng tâm cho
		# từng bounding box

		# duyệt qua từng bounding box trong rects
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# lấy ra tọa độ trọng tâm của box và lưu vào inputCentroids
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# nếu chưa theo dõi bất kì đối tượng nào thì tiến hành đăng ký đối tượng mới
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# ngược lại, cập nhật lại các đối tượng đã được đăng ký rồi
		else:
			# lấy các giá trị ID và giá trị trọng tâm
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# tính khoảng cách của các trọng tâm của các đối tượng đã theo dõi và các đối tượng chưa theo dõi
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# tìm giá trị min của mỗi hàng và sắp xếp cho index của giá trị min lên đầu hàng
			rows = D.min(axis=1).argsort()   #argsort dung de sap xep cac chi muc

			# thực hiện tương tự đối với các cột, sắp xếp dựa trên hàng
			cols = D.argmin(axis=1)[rows]

			# tạo 2 set chứa các index đã sử dụng
			usedRows = set()
			usedCols = set()

			# duyệt qua các tập index hàng và cột
			for (row, col) in zip(rows, cols):
				# chỉ mục nào đã được sử dụng thì bỏ qua
				if row in usedRows or col in usedCols:
					continue

				# nếu chưa dùng thì lấy giá trị ID của index hiện tại,set làm trọng tâm mới: row là ID, cột là trọng tâm
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# cập nhật lại các chỉ mục hàng và cột đã sử dụng
				usedRows.add(row)
				usedCols.add(col)

			# xử lý các centroid indexs chưa được dùng
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# Nếu số lượng các trọng tâm của đối tượng trong objectCentroids lớn hơn hoặc bằng số lượng các trọng tâm trong inputCentroids 
			# inputCentroids chứa các trọng tâm của các bounding box trong frame, còn objectCentroids chứa các trọng tâm đã đăng ký rồi
			# => có các đối tượng đã bị mất track.
			if D.shape[0] >= D.shape[1]:
				# duyệt qua các index chưa được dùng, lấy giá trị ID và tăng giá trị disappeared lên
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# Hủy đăng ký đối tượng trong khi disappeared đạt tối đa
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# ngược lại, D.shape(objectCentroids) < D.shape(inputCentroids), số lượng inputCentroids lớn hơn số lượng objectCentroids đã tồn tại
			# => có thêm đối tượng mới, đăng ký theo dõi các đối tượng mới
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# trả về các đối tượng đã được theo dõi
		return self.objects