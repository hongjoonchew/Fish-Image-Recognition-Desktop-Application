import imgaug
import argparse
import cv2
import os
import json

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
	pass

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

	annotation = open(args.annotation_path)
	jsonAnnotation = json.load(annotation)

# load bounding boxes with json encoding library
annotation = open(args.annotation_path)
jsonAnnotation = json.load(annotation)


for image in jsonAnnotation:
	filename = image["filename"]
	images[filename.split('/')[2]].append(image["annotations"])		

	# load bounding boxes with json encoding library
 	# apply bounding boxes to images	

result = processImages(images, delta)
	
#cv2.imwrite(outputFilename, result)
# save bounding boxes in annotation files
