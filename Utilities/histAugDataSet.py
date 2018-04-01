# Pisces Project
# Last modified: 2/14/18
# Use: Requires Python 2.x 


import numpy as np
import cv2
import argparse
import os

def processImageColor(img):
	# img = image
	#Z = img.reshape((-1,3))

	# # convert to np.float32
	#Z = np.float32(Z)

	# # define criteria, number of clusters(K) and apply kmeans()
	# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	# ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

	# # Now convert back into uint8, and make original image
	# center = np.uint8(center)
	# res = center[label.flatten()]
	# res2 = res.reshape((img.shape))
		
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		# equalize the histogram of the Y channel
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
		# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	

	#else:
	#	img_output = cv2.equalizeHist(img)
	#	print(img.shape)
	
	
	return img_output
	
def processImageBW(img):
	return(cv2.equalizeHist(img))
	

parser = argparse.ArgumentParser(description='Reduce the number of colors in an image with k-means clustering')

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

# Creates output folder if folder doesn't exist
if not os.path.exists(directoryPrefix + "output"):
	os.mkdir(directoryPrefix + "output")
	
imgCount = 0
outputFolderPath = directoryPrefix + "output/"
 
for file in filepaths:
	fileName, extension = file.split(".")
	img = cv2.imread(directoryPrefix + file ,0)
	
	result = processImageBW(img)
	cv2.imwrite(outputFolderPath + fileName + ".jpg", result)
	imgCount += 1
	
cv2.waitKey(0)
cv2.destroyAllWindows()


