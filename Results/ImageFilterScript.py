import cv2
import numpy as np
import sys
import os
import glob
from pathlib import Path
from color_transfer import color_transfer

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "./Results/")

filelist = glob.glob("../Samples/*.jpg",recursive=True)
print(path)
print(filelist)

height = 50
width = 50
murky_water  = "Murkywater.jpg"
water_img = cv2.imread(murky_water, 1)
water_img = cv2.resize(water_img, (height,width))
for file in filelist:
	try:
		img = cv2.imread(file, 1) # scrolls through all the fish images as of late.
		small = cv2.resize(img, (height,width)) #changes images into a 400px by 250px size.
		transfer = color_transfer(water_img,small)
		cv2.imshow('Fish filter img',transfer)
		cv2.imwrite(os.path.basename(file), transfer)
		cv2.waitKey(0)
		cv2.destroyAllWindows
	except IOError as exc:
		if exc.errno != errno.EISDIR:
			raise

