import os
import shutil

file_directory = "/n09376526/"
directory = os.getcwd() + file_directory
labels = directory + "/labels/"
imagelist = os.getcwd() + "/dontcare_dataset/train/images/"

for filename in os.listdir(directory):
   root, ext = os.path.splitext(filename)
   name = root + ".JPEG"
   if name in os.listdir(imagelist):
   	  shutil.copy(imagelist + name, os.getcwd() + "/xml images/")