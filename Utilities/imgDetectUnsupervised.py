#
# File info 
#
#
#

import numpy as np
import cv2
import argparse
import os

def processImage(K, image):
	img = image
	Z = img.reshape((-1,3))

	# convert to np.float32
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	
	return res2
	

parser = argparse.ArgumentParser(description='Reduce the number of colors in an image with k-means clustering')

parser.add_argument('k', type=int, help='Number of classes for k-means clustering')
parser.add_argument('pathname', metavar="path_to_file", help='File to be processed')
					
args = parser.parse_args()

directoryPrefix = './'

filepaths = []

if os.path.isdir(args.pathname):
	directoryPrefix = args.pathname + '/'
	# Recursively walks through the directory structure and creates two lists. One for directories
	# and one for file names.
	for (dirpath, dirnames, filenames) in os.walk(args.pathname):
		filepaths.extend(filenames)
		break 
	
	# Creates output folder if folder doesn't exist
	if not os.path.exists(directoryPrefix + "output"):
		os.mkdir(directoryPrefix + "output")
else:
	filepaths = [ args.pathname ]
	
imgCount = 0
outputFolderPath = directoryPrefix + "output/"

for file in filepaths:
	img = cv2.imread(directoryPrefix + file)
	
	result = processImage(args.k, img)
	
	cv2.imwrite(outputFolderPath +  str(imgCount) + "_" + str(args.k) + ".jpg", result)

	imgCount += 1
	
cv2.waitKey(0)
cv2.destroyAllWindows()


