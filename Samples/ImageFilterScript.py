import cv2
import numpy as np
import sys
import os
import glob
from pathlib import Path

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "../Rainbow trout-steelhead/*.jpg")
filelist = glob.glob('**/*.jpg',recursive=True)
for file in filelist:
	try:
		img = cv2.imread(file, 0) # scrolls through all the fish images as of late.
		cv2.imshow('image',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows
	except IOError as exc:
		if exc.errno != errno.EISDIR:
			raise

