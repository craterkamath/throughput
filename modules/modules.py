import os
from yolokeras import yolo
from PIL import Image
import numpy as np

def model_classifier(image):
	new_im = Image.fromarray(image)
	if not os.path.exists('yolokeras/images'):
		os.makedirs('yolokeras/images')
	new_im.save("yolokeras/images/image.png")
	IMG_PATH = "yolokeras/images/image.png"
	yolo.main(IMG_PATH)


# img_path = '/home/nikhil/Desktop/hackathon/hack/images/'
# for path in os.listdir(img_path):	
#  	input_image = Image.open(os.path.join(img_path,path))
#  	np_im = np.array(input_image)
#  	model_classifier(np_im)
