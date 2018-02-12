import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description='Reduce the number of colors in an image with k-means clustering')

parser.add_argument('k', type=int, help='Number of classes for k-means clustering')
parser.add_argument('filename', metavar="path_to_file", help='File to be processed')
					
args = parser.parse_args()

img = cv2.imread(args.filename)
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('res2',res2)
cv2.imwrite('result.jpg', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
