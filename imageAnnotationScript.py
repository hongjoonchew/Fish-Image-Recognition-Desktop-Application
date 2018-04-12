import sys
import cv2
import numpy as np
import Tkinter, tkFileDialog
import os
import datetime
import json
import argparse
from objectDetectImage import detectImage

CONFIDENCE_MIN = 0.2

CLASSES = ["DontCare", "Car"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES),3))

# net = cv2.dnn.readNetFromCaffe("../deploy.prototxt1.txt", "../snapshot_iter_2538.caffemodel")

# Setting up using DetectNet Model with caffe
model_file = "/home/sagis/Desktop/epoch/xdeploy.prototxt"
pretrained_model = "/home/sagis/Desktop/epoch/snapshot_iter_27216.caffemodel"

root = Tkinter.Tk()
filez = tkFileDialog.askopenfilenames(parent=root,title='Choose a file')
#print root.tk.splitlist(filez)

fileList = root.tk.splitlist(filez)

size = len(fileList)

#print size

parser = argparse.ArgumentParser(description='Annotation Script using DetectNet.')
parser.add_argument("--output_dir",type=str, help="Output Directory of the Annotations")


output_directory = sys.argv[1]   # IMPORTANT THIS NEEDS TO HAVE THE BASE FILE LOCATION this is where the label and images directories will be created and files saved\
                            # for example C:/UserName/Desktop/test/
base_file_name = "Annotations"  # create file name base for screenshot and annotation file, this could be set to the current data/time

label_directory = output_directory + "labels/"

if not os.path.exists(label_directory):
    os.makedirs(label_directory)

file_name = label_directory + base_file_name + ".json"


openFile = open(file_name, "w+")


for x in fileList:

    #Use Caffe instead
    detections = detectImage(model_file, pretrained_model,x)
    """blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    """
    
    """
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]


    if confidence > CONFIDENCE_MIN:
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box

        bbox = {"top":startX, "left":startY, "bottom":endX, "right":endY}

        openFile.write(json.dumps(bbox))#writes json object in Annotation.json file

        openFile.write(",\n")


        (startX, startY, endX, endY) = box.astype("int")

        label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
        object_class = CLASSES[int(detections[0, 0, i, 1])]
        cv2.rectangle(img, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLORS[idx], 2)
        """
    for bbox in detections:
               