import sys
import cv2
import numpy as np
import Tkinter,tkFileDialog
import os
import datetime

CONFIDENCE_MIN = 0.2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES),3))
 
net = cv2.dnn.readNetFromCaffe("../MobileNetSSD_deploy.prototxt.txt", "../MobileNetSSD_deploy.caffemodel")



root = Tkinter.Tk()
filez = tkFileDialog.askopenfilenames(parent=root,title='Choose a file')
print root.tk.splitlist(filez)

fileList = root.tk.splitlist(filez)

size = len(fileList)

print size


def generate_annotation_file(file_name, detections, img_w, img_h):
    #current hard coded values, to make these variable change DETECTION_STRING_FORMAT
    truncated = '0'
    occluded = '3'
    alpha = '0'
    dimensions = ['0','0','0']
    location = ['0','0','0']
    rotation_y = '0'
    score = '0'

    DETECTION_STRING_FORMAT = "{} " + \
                              truncated + " " +\
                              occluded + " " +\
                              alpha + " " +\
                              "{:02.1f} {:02.1f} {:02.1f} {:02.1f} "  +\
                              dimensions[0] + " " + dimensions[1] + " " + dimensions[2] + " " +\
                              location[0] + " " + location[1] + " " + location[2] + " " +\
                              rotation_y + " "+\
                              score + "\n"


    with open(file_name, 'a+') as annotation_file:
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE_MIN:
                object_class = CLASSES[int(detections[0, 0, i, 1])]
                box = detections[0, 0, i, 3:7] * np.array([img_w, img_h, img_w, img_h])
                (startX, startY, endX, endY) = box.astype("int")
                bbox = {"top":startX, "left":startY, "bottom":endX, "right":endY}

                annotation_file.write(DETECTION_STRING_FORMAT.format(object_class, bbox["top"], bbox["left"], bbox["bottom"], bbox["right"]))

def create_screen_capture(img,detections,w,h):
    now = datetime.datetime.now()
    output_directory = "C:/Users/IBM_ADMIN/Desktop/"   # IMPORTANT THIS NEEDS TO HAVE THE BASE FILE LOCATION this is where the label and images directories will be created and files saved\
                            # for example C:/UserName/Desktop/test/
    base_file_name = now.strftime(
        "%Y-%m-%d_%H%M")  # create file name base for screenshot and annotation file, this could be set to the current data/time
    
    label_directory = output_directory + "labels/"
    
    if not os.path.exists(label_directory):
        os.makedirs(label_directory)
    generate_annotation_file(label_directory + base_file_name + ".txt", detections, w, h)


for x in fileList:

    img = cv2.imread( x, cv2.IMREAD_COLOR )

    (h,w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

    img_copy = img.copy() #create a copy without bounding boxes for screen capture

    if confidence > CONFIDENCE_MIN:
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
        cv2.rectangle(img, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLORS[idx], 2)
        create_screen_capture(img_copy,detections,w,h)


