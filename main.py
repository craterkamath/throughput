import numpy as np
import cv2

cap = cv2.VideoCapture('test.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(100, 25, False)
sample_tracker = cv2.TrackerCSRT_create()
INIT = True
MAX_COUNT = 25

class Node:
	def __init__(self, data, masks, boxes, tracker):
		self.data = data
		self.masks = masks
		self.boxes = boxes
		self.tracker = tracker

	def addFrame(self, frame, mask, box):
		self.data.append(frame)
		self.masks.append(mask)
		self.boxes.append(box)

	def removeFrame(self, idx):
		out_frame = self.data.pop(idx)
		out_mask = self.masks.pop(idx)
		out_box = self.boxes.pop(idx)
		return out_frame, out_mask, out_box

	def getLatestCentroid(self):
		x, y, w, h = self.boxes[0]
		return (x+w/2, y+h/2)

	def getClosestBoxandAdd(self, frame, bboxes, mask):
		success, box = self.tracker.update(frame)

		if success:
			if bboxes:
				min_value = 999999
				min_idx = -1
				for idx in range(len(bboxes)):
					dist_norm = np.linalg.norm(np.array(box) - np.array(bboxes[idx]))
					if dist_norm < min_value:
						min_idx = idx
						min_value = dist_norm
				self.addFrame(frame, mask, bboxes[min_idx])
				return min_idx
			else:
				self.addFrame(frame, mask, box)
				return -1
		else:
			return -1


object_q = []
active_q = []

while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:
		fg_mask = fgbg.apply(frame)
		_ , fg_mask = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)
		kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3))


		fg_mask_er = cv2.dilate(fg_mask, kernel = kernel, iterations = 3)
		fg_mask_er = cv2.erode(fg_mask_er, kernel = kernel, iterations = 3)
		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		output_frame_er = cv2.bitwise_and(frame, frame, mask = fg_mask_er)
		
		img_mod, contours, heirarchy = cv2.findContours(fg_mask_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		convex_hulls = []
		hull_boxes = []
		centroids = []

		for contour in contours:
			hull = cv2.convexHull(contour)
			rect = cv2.boundingRect(hull)
			x, y, w, h = rect


			if(cv2.contourArea(contour) > 180):
				convex_hulls.append(hull)
				hull_boxes.append(rect)
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
				centroids.append((x+w/2, y+h/2))


		if INIT:
			INIT = False
			for box in hull_boxes:
				tracker = cv2.TrackerCSRT_create()
				tracker.init(frame, box)
				node = Node([frame], [output_frame_er], [box], tracker)
				object_q.append(node)
			else:
				for idx in range(len(object_q)):
					return_idx = object_q[idx].getClosestBoxandAdd(frame, hull_boxes, fg_mask_er)
					if return_idx != -1:
						hull_boxes.pop(return_idx)
				while hull_boxes:
					tracker = cv2.TrackerCSRT_create()
					tracker.init(frame, hull_boxes[0])
					node = Node([frame], [output_frame_er], [hull_boxes[0]], tracker)
					object_q.append(node)
					hull_boxes.pop(0)


		cv2.imshow("input", frame)
		cv2.moveWindow("input", 100, 100)

		cv2.imshow("Moving Objects", output_frame_er)
		cv2.moveWindow("Moving Objects", 700, 200)

		cv2.imshow("contours", img_mod)
		cv2.moveWindow("contours", 300, 500)

		cv2.waitKey(1)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()
