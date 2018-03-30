import os
import json
from PIL import Image
import random
import shutil
from XML import minidom

file_directory = "n09376526"
directory = os.getcwd() + file_directory

for file_name in file_directory:
   print(file_name)
