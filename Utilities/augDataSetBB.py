import imagaug
import argparse

parser = argparse.ArgumentParser(description='Augments image data with the following transformations:
crop random
flip left right
flip top down
pad
Gaussian blur
contrast normalization
greyscale')

parser.add_argument('n', type=int, help='Number of images to generate')
parser.add_argument('path', type=str, help='Path for folder to images')

args = parser.parse_args()



#
# parse command line arguments: nOutputImages, inputDirectory, outputDirectory
# open images in 
# add bounding boxes 
# create bounding boxes 
# create transformations
# apply transformations
# save images
# save bounding boxes in annotation file
#
# transformations: crop random, pad, flip
