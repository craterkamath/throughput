import sys
import numpy as np
import cv2


cap = cv2.VideoCapture(sys.argv[1])

ret ,frame = cap.read()

background =  []

avg1 = np.float32(frame)
avg2 = np.float32(frame)

while(True):
    ret,frame = cap.read()

    if not ret:
    	break
    cv2.accumulateWeighted(frame,avg1,0.1)
    cv2.accumulateWeighted(frame,avg2,0.01)

    res1 = cv2.convertScaleAbs(avg1)
    res2 = cv2.convertScaleAbs(avg2)
    background = res2

cap = cv2.VideoCapture(sys.argv[1])
fgbg = cv2.createBackgroundSubtractorMOG2(100, 50, False)
# trackers = cv2.MultiTracker_create()
INIT = True
MAX_COUNT = 25
GEN_IMAGES = False
GEN_COUNT = 1000
SHOW_OUTPUT = True
ptr = 0
object_q = []
active_q = []
SCALE = 0.75


def doOverlap(l1x, l1y , r1x, r1y, l2x, l2y, r2x, r2y): 
      
    if(l1x > r2x or l2x > r1x): 
        return False
  
    if(l1y < r2y or l2y < r1y): 
        return False
  
    return True

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




while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:
		x, y, d = frame.shape
		frame = cv2.resize(frame, (int(y * SCALE), int(x * SCALE)))
		background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
		frame_cpy = frame.copy()
		fg_mask = fgbg.apply(frame)
		_ , fg_mask_er = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)
		kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3))


		fg_mask_er = cv2.dilate(fg_mask_er, kernel = kernel, iterations = 3)
		fg_mask_er = cv2.erode(fg_mask_er, kernel = kernel, iterations = 3)
		# fg_mask_er = fg_mask
		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		output_frame_er = cv2.bitwise_and(frame, frame, mask = fg_mask_er)
		
		img_mod, contours, heirarchy = cv2.findContours(fg_mask_er, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		convex_hulls = []
		hull_boxes = []
		centroids = []

		for contour in contours:
			hull = cv2.convexHull(contour)
			rect = cv2.boundingRect(hull)
			x, y, w, h = rect
			
			if GEN_IMAGES and ptr < GEN_COUNT:
				cv2.imwrite("image{0}.png".format(ptr), frame[y:y+h, x:x+w])
				ptr += 1

			if(cv2.contourArea(contour) > 300):
				convex_hulls.append(hull)
				hull_boxes.append(rect)
				cv2.rectangle(frame_cpy, (x, y), (x+w, y+h), (0, 255, 0), 2)
				centroids.append((x+w/2, y+h/2))


		if INIT:
			INIT = False
			for box in hull_boxes:
				tracker = cv2.TrackerKCF_create()
				tracker.init(frame, box)
				node = Node([frame], [fg_mask_er], [box], tracker)
				object_q.append(node)
		else:
			for idx in range(len(object_q)):
				return_idx = object_q[idx].getClosestBoxandAdd(frame, hull_boxes, fg_mask_er)
				if return_idx != -1:
					hull_boxes.pop(return_idx)
			while hull_boxes:
				tracker = cv2.TrackerCSRT_create()
				tracker.init(frame, hull_boxes[0])
				node = Node([frame], [fg_mask_er], [hull_boxes[0]], tracker)
				object_q.append(node)
				hull_boxes.pop(0)

		if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		if SHOW_OUTPUT:
			cv2.imshow("input", frame_cpy)
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
output_video = []
frame_number = 0

k = 0
# for obj in object_q:
# 	k += 1
# 	for img in obj.data:
# 		cv2.imshow("Object {0}".format(k), img)
# 		cv2.waitKey(15)
# 	cv2.waitKey(0)


# import pdb; pdb.set_trace()

while(object_q or active_q):
	while len(active_q) < MAX_COUNT and object_q:
		active_q.append(object_q.pop(0))

	output_frame = background
	frame_number += 1

	try:
		for i in range(len(active_q)):
			collide = False
			x1, y1, w1, h1 = active_q[i].boxes[0]
			for j in range(i):
				x2, y2, w2, h2 = active_q[j].boxes[0]
				if(doOverlap(x1, y1, x1+w1, y1+h1, x2, y2, x2+w2, y2+h2)):
					collide = True
					break

			if not collide:
				blend_frame, blend_mask, blend_box = active_q[i].removeFrame(0)
				overlap_mask = np.zeros(output_frame.shape, dtype = np.uint8)
				cv2.rectangle(overlap_mask, (int(blend_box[0]), int(blend_box[1])), (int(blend_box[0]+blend_box[2]), int(blend_box[1]+blend_box[3])), (255, 255, 255), -1)

				# print(blend_frame.dtype)
				# print(blend_mask.dtype)
				# print(blend_frame.shape)
				# print(blend_mask.shape)


				# cv2.imshow("Frame", blend_frame)
				# cv2.imshow("blend", blend_mask)
				# cv2.waitKey(0)


				frame_content = cv2.bitwise_and(blend_frame, blend_frame, mask = blend_mask)

				# print(frame_content.dtype)
				# print(overlap_mask.dtype)
				# print(frame_content.shape)
				# print(overlap_mask.shape)


				# cv2.imshow("Frame Content", frame_content)
				# cv2.imshow("Overlap Mask", overlap_mask)
				# cv2.waitKey(0)

				# output_object = cv2.bitwise_and(frame_content, frame_content,mask = overlap_mask)


				# print(blend_frame.dtype)
				# print(blend_mask.dtype)
				# print(overlap_mask.dtype)
				# print(blend_frame.shape)
				# print(blend_mask.shape)
				# print(overlap_mask.shape)
				# cv2.imshow("Blend Frame", blend_frame)
				# cv2.imshow("Blend Mask", blend_mask)
				# cv2.imshow("Overlap Mask", overlap_mask)
				# cv2.waitKey(0)

				overlap_mask_1d, _, _ = cv2.split(overlap_mask)
				# import pdb; pdb.set_trace()
				output_mask = cv2.bitwise_and(blend_mask, blend_mask, mask = overlap_mask_1d)
				output_mask_inv = cv2.bitwise_not(output_mask)

				# cv2.imshow("Frame", output_frame)
				# cv2.imshow("Object", output_object)
				# cv2.waitKey(0)

				output_object = cv2.bitwise_and(frame_content, frame_content, mask = output_mask)
				output_frame = cv2.addWeighted(output_frame, 1.0, output_object, 1, 0.0)
	except:
		print("Error in Frame: {0}".format(frame_number))

	output_video.append(output_frame)

	for obj in active_q:
		if(len(obj.data) == 0):
			active_q.remove(obj)


for img in output_video:
	cv2.imshow("Output", img)
	cv2.waitKey(0)

# out = cv2.VideoWriter('out_video.avi', -1, 21, output_video[0].shape[:2])
# for img in output_video:
# 	out.write(img)
# out.release()