import imgaug as ia
from imgaug import augmenters as iaa
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


def readData(input_path, output_path, annotation_path):
	directoryPrefix = './'

	filepaths = []

	outputFolderPath =  output_path

	if os.path.isdir(input_path):
		directoryPrefix = input_path + '/'
		# Recursively walks through the directory structure and creates two lists. One for directories
		# and one for file names.
		for (dirpath, dirnames, filenames) in os.walk(input_path):
			filepaths.extend(filenames)
			break 
		
		# Creates output folder if folder doesn't exist
		if not os.path.exists(outputFolderPath):
			os.mkdir(outputFolderPath)
	else:
		filepaths = [ input_path ]


	# Dictionary stores {image_name, [image, bounding_box]}
	images = {}

	print("Reading images..")
	for file in filepaths:
		img = cv2.imread(directoryPrefix + file)
		print(file)
		images[file] = [img]
	print("Done.") 

	# load bounding boxes with json encoding library
	annotation = open(annotation_path)
	jsonAnnotation = json.load(annotation)
	annotation.close()

	print("Loading annotation file...")
	for entry in jsonAnnotation:
		# Filename format is ../1-50/Image00--.jpg
		tokens = entry["filename"].split('/')
	 	
		if(len(tokens) == 1):
			filename = entry["filename"].split("\\")[-1]
		else:
			filename = entry["filename"].split("/")[-1]
	
		image = images[filename][0]
		boundingBoxes = parseBoundingBoxes(entry["annotations"], image.shape)
		images[filename].append(boundingBoxes)		

		# load bounding boxes with json encoding library
	 	# apply bounding boxes to images	
	print("Done")
	return(images)



def processImages(images, delta):
	# apply transformations
	imgList = []
	boundingBoxes = []
	print("starting augmentation")
	for imageName, data in images.items():
		imgList.append(data[0])
		boundingBoxes.append(data[1])

	image_aug = delta.augment_images(imgList)
	bbs_aug = delta.augment_bounding_boxes(boundingBoxes)
	print("done augmenting")
	
	return image_aug, bbs_aug

def writeData(augImages, augBBoxes, outputPath):
	jsonOut = []
	for index in range(len(augImages)):
		# generate file name
		outFileName = "Image" + str(index) + '_' +  datetime.utcnow().isoformat() + '.jpg'
		outFileName = outFileName.replace(":", "_")
		# generate json entry for file
		annotations = []

		for bb in augBBoxes[index].bounding_boxes:
			annotations.append({"class":"rect", "height": bb.y2 - bb.y1, "width": bb.x2 - bb.x1, "x": bb.x1, "y": bb.y1 })
		jsonOut.append({"annotations":annotations, "class":"image", "filename":outFileName})
		# write image to output directory
		imagePath = outputPath + "/" + outFileName
		try:
			cv2.imwrite(imagePath, augImages[index])
		except IndexError:
			print("Error accessing image data for " + imagePath)
		except KeyError:
			print("Key error. Image not written")
			print(imagePath)
	# construct json)
	# parse as json with json.dumps()
	annotationFile = open(outputPath + '/annotations.json', "w")
	# write to annotation file
	annotationFile.write(json.dumps(jsonOut, sort_keys=True, indent=4, separators=(',', ': ')))


def parseBoundingBoxes(annotations, imageShape):
	# input: annotation corresponding to a single bounding box
	# output: BoundingBox object
	boundingBoxes = []
	for annotation in annotations:
		xTopLeft = annotation["x"]
		yTopLeft = annotation["y"]
		xBottomRight = xTopLeft + annotation["width"]
		yBottomLeft = yTopLeft + annotation["height"]

		boundingBox = ia.BoundingBox(x1=xTopLeft,y1=yTopLeft,x2=xBottomRight, y2=yBottomLeft)
		
		boundingBoxes.append(boundingBox)
	
	bbs = ia.BoundingBoxesOnImage(boundingBoxes, shape=imageShape)
	return(bbs)






# parser.add_argument('n', type=int, help='Number of images to generate')
parser.add_argument('input_path', type=str, help='Path to images')
parser.add_argument('output_path', type=str, help='Path for output images')
parser.add_argument('annotation_path', type=str, help='Path for annotation file')

args = parser.parse_args()


ia.seed(1)

# defines transformations to apply to images
seq = iaa.Sequential([
	iaa.Grayscale(alpha=1.0)
], random_order=False)

# Make our sequence deterministic.
# We can now apply it to the image and then to the BBs and it will
# lead to the same augmentations.
# IMPORTANT: Call this once PER BATCH, otherwise you will always get the
# exactly same augmentations for every batch!
seq_det = seq.to_deterministic()

print("Augmenting images...")
augImages, augBBoxes = processImages(readData(args.input_path, args.output_path, args.annotation_path), seq_det)
print("Done.")

print("Saving images...")
writeData(augImages, augBBoxes, args.output_path)
print("Done.")
