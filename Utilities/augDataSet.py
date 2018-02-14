import argparse
import Augmentor

parser = argparse.ArgumentParser(description='Augments image data for a deep learning model')

parser.add_argument('n', type=int, help='Number of images to generate')
parser.add_argument('path', type=str, help='Path for folder to images')

args = parser.parse_args()

pipe = Augmentor.Pipeline(args.path)

NUM_IMAGES = args.n

MAX_DEGREES = 10

pipe.rotate(probability=0.7, max_left_rotation=MAX_DEGREES, max_right_rotation=MAX_DEGREES)

pipe.skew(probability=0.5)

pipe.flip_left_right(probability=1)

pipe.histogram_equalisation(probability=1)

pipe.sample(NUM_IMAGES)
