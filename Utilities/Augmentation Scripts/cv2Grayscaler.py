import os
import cv2
import shutil

directory = os.getcwd()

for filename in os.listdir(directory):
   print(filename)
   image = cv2.imread(filename,0)
   cv2.imwrite(filename, image)
