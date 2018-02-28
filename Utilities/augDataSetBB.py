import imgaug
import argparse
import cv2
import os
import json
from datetime import datetime

parser = argparse.ArgumentParser(description="""Augments image data with the following transformations:\n
crop random\n
flip left right\n
flip top down\n
pad\n
Gaussian blur\n
contrast normalization\n
greyscale\n""")


def processImages(images, delta):
	# apply transformations
	result = []
	for key, value in images.items():
		result.append(value)
	return(result)

def writeData(images, imageOutPath, annotationOutPath):
	jsonOut = []
	for key, value in images:
		# generate file name
		outFileName = key + '_' +  datetime.isoformat(timespec='microsecond')
		# generate json entry for file
		try:
			jsonOut.append({"annotation":value[1], "class":"image", "filename":outFileName})
		except IndexError:
			print("Error accesing annotations for " + key)
		# write image to output directory
		try:
			cv2.imwrite(imageOutPath + outFileName, images[0])
		except IndexError:
			print("Error accessing image data for " + key)
	# construct json)
	# parse as json with json.dumps()
	annotationFile = open(annotationOutPath, "w")
	# write to annotation file
	annotationFile.write(json.dumps(jsonOut))


parser.add_argument('n', type=int, help='Number of images to generate')
parser.add_argument('input_path', type=str, help='Path to images')
parser.add_argument('output_path', type=str, help='Path for output images')
parser.add_argument('annotation_path', type=str, help='Path for annotation file')

args = parser.parse_args()

directoryPrefix = './'

filepaths = []

outputFolderPath = directoryPrefix + args.output_path

if os.path.isdir(args.input_path):
	directoryPrefix = args.input_path + '/'
	# Recursively walks through the directory structure and creates two lists. One for directories
	# and one for file names.
	for (dirpath, dirnames, filenames) in os.walk(args.input_path):
		filepaths.extend(filenames)
		break 
	
	# Creates output folder if folder doesn't exist
	if not os.path.exists(outputFolderPath):
		os.mkdir(outputFolderPath)
else:
	filepaths = [ args.input_path ]

# define transformations to apply to images


# Dictionary stores <image_name, [image, bounding_box]>
images = {}

for file in filepaths:
	# bug: loads annotation file as well as images
	# possible solutions: store annotations separate from images
	# Possible file structure: /images /annotations
	img = cv2.imread(directoryPrefix + file)
	
	images[file] = [img] 

# load bounding boxes with json encoding library
annotation = open(args.annotation_path)
jsonAnnotation = json.load(annotation)


for image in jsonAnnotation:
	filename = image["filename"]
	images[filename.split('/')[2]].append(image["annotations"])		

	# load bounding boxes with json encoding library
 	# apply bounding boxes to images	

result = processImages(images, 0)


# writeImages()
