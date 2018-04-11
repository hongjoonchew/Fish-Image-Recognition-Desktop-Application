# Pisces Project
# Last modified: 2/14/18
# Use: Requires Python 2.x 


import numpy as np
import cv2
import argparse
import os
	
def processImage(img):
	return(img)
	

parser = argparse.ArgumentParser(description='Transform a color input file to grayscale')

parser.add_argument('pathname', metavar="path_to_folder", help='Folder to be processed')
					
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
	
	result = processImage(img)
	cv2.imwrite(outputFolderPath + fileName + ".jpg", result)
	imgCount += 1
	
cv2.waitKey(0)
cv2.destroyAllWindows()


