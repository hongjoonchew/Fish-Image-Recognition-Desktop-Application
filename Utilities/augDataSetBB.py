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


def processImages(images, delta):
	# apply transformations
	imgList = []
	boundingBoxes = []
	print("starting augmentation")
	for imageName, data in images.items():
		imgList.append(data[0])
		boundingBoxes.append(data[1])
	print("done augmenting")
	image_aug = delta.augment_images(imgList)
	bbs_aug = delta.augment_bounding_boxes(boundingBoxes)
	
	return image_aug, bbs_aug

def writeData(augImages, augBBoxes, outputPath):
	jsonOut = []
	for index in range(len(augImages)):
		# generate file name
		outFileName = "Image" + str(index) + '_' +  datetime.utcnow().isoformat() + '.jpg'
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

directoryPrefix = './'

filepaths = []

outputFolderPath =  args.output_path

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


# Dictionary stores {image_name, [image, bounding_box]}
images = {}

print("Reading images..")
for file in filepaths:
	# bug: loads annotation file as well as images
	# possible solutions: store annotations separate from images
	# Possible file structure: /images /annotations
	img = cv2.imread(directoryPrefix + file)
	print(file)
	images[file] = [img]
print("Done.") 

# load bounding boxes with json encoding library
annotation = open(args.annotation_path)
jsonAnnotation = json.load(annotation)

print("Loading annotation file...")
for entry in jsonAnnotation:
	# Filename format is ../1-50/Image00--.jpg
	filename = entry["filename"].split('/')[-1]
	
	image = images[filename][0]
	boundingBoxes = parseBoundingBoxes(entry["annotations"], image.shape)
	images[filename].append(boundingBoxes)		

	# load bounding boxes with json encoding library
 	# apply bounding boxes to images	
print("Done")


ia.seed(1)

# defines transformations to apply to images
seq = iaa.Sequential([
	iaa.Fliplr(0.5), # horizontal flip
	# Small gaussian blur with random sigma between 0 and 0.5
	# But only about 50% of the images are blurred
	iaa.Sometimes(0.5, 
		iaa.GaussianBlur(sigma=(0, 0.5))
	),
	# Strengthen or weaken the contrast in each image.
	iaa.ContrastNormalization((0.75, 1.5)),
	# Add gaussian noise.
	# For 50% of all images, we sample the noise once per pixel.
	# For the other 50% of all images, we sample the noise per pixel AND
	# channel. This can change the color (not only brightness) of the
	# pixels.
	iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
	# Make some images brighter and some darker.
	# In 20% of all cases, we sample the multiplier once per channel,
	# which can end up changing the color of the images.
	iaa.Multiply((0.8, 1.2), per_channel=0.2),
	# Apply affine transformations to each image.
	# Scale/zoom them, translate/move them, rotate them and shear them.
	iaa.Affine(
        	scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        	translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        	rotate=(-10, 10),
    	)
], random_order=True)

# Make our sequence deterministic.
# We can now apply it to the image and then to the BBs and it will
# lead to the same augmentations.
# IMPORTANT: Call this once PER BATCH, otherwise you will always get the
# exactly same augmentations for every batch!
seq_det = seq.to_deterministic()

print("Augmenting images...")
augImages, augBBoxes = processImages(images, seq_det)
print("Done.")

print("Saving images...")
writeData(augImages, augBBoxes, outputFolderPath)
print("Done.")
