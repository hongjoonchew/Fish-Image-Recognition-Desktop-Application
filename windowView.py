from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import numpy as np
import os
import datetime
import os
import platform
import subprocess


output_directory = "/"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES),3))

CONFIDENCE_MIN = 0.2

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

def show_webcam_with_model(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)

        (h,w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        img_copy = img.copy() #create a copy without bounding boxes for screen capture

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
        
            if confidence > CONFIDENCE_MIN:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
                cv2.rectangle(img, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLORS[idx], 2)
        cv2.imshow('Project Pisces Cam', img)

        key = cv2.waitKey(1) & 0xFF
        # if the `p` key was pressed, screen capture
        if key == ord('p'):
            create_screen_capture(img_copy,detections,w,h)

        # if the `q` key was pressed, break from the loop
        if key == ord('q'):
            print("Q pressed")
    cv2.destroyAllWindows()


def processVideo(self, filepath):
    cap = cv2.VideoCapture(filepath)

    while (cap.isOpened()):
        ret, img = cap.read()
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        img_copy = img.copy()  # create a copy without bounding boxes for screen capture

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE_MIN:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(img, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLORS[idx], 2)

        cv2.imshow('Project Pisces Cam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


class Window(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setupUi(self)
    
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setMinimumSize(QtCore.QSize(500, 450))
        Form.setMaximumSize(QtCore.QSize(500, 450))
        Form.setStyleSheet("background-color: #632733")
        self.runButton = QtWidgets.QPushButton(Form)
        self.runButton.setGeometry(QtCore.QRect(15, 400, 150, 40))
        self.runButton.setObjectName("runButton")
        self.runButton.setStyleSheet("background-color: #FFFFFF")
        self.inputVideoButton = QtWidgets.QPushButton(Form)
        self.inputVideoButton.setGeometry(QtCore.QRect(175, 400, 150, 40))
        self.inputVideoButton.setObjectName("inputVideoButton")
        self.inputVideoButton.setStyleSheet("background-color: #FFFFFF")
        self.setDefaultDirectoryButton = QtWidgets.QPushButton(Form)
        self.setDefaultDirectoryButton.setGeometry(QtCore.QRect(335, 400, 150, 40))
        self.setDefaultDirectoryButton.setObjectName("setDefaultDirectoryButton")
        self.setDefaultDirectoryButton.setStyleSheet("background-color: #FFFFFF")
        self.initButtons(self.runButton, self.inputVideoButton, self.setDefaultDirectoryButton)
        self.imageLabel = QtWidgets.QLabel(Form)
        self.imageLabel.setGeometry(QtCore.QRect(20, 10, 460, 380))
        self.imageLabel.setObjectName("imageLabel")
        img = QtGui.QPixmap("logo.png")
        self.imageLabel.setPixmap(img)
        self.imageLabel.setScaledContents(True)
        self.imageLabel.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.imageLabel.show()
        
        
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        self.show()
    
    def initButtons(self, runButton, inputVideoButton, setDefaultDirectoryButton):
        runButton.clicked.connect(self.runButtonPressed)
        inputVideoButton.clicked.connect(self.inputVideoButtonPressed)
        setDefaultDirectoryButton.clicked.connect(self.setDefaultDirectoryButtonPressed)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Project Pisces"))
        self.runButton.setText(_translate("Form", "Run Model"))
        self.inputVideoButton.setText(_translate("Form", "Input Video"))
        self.setDefaultDirectoryButton.setText(_translate("Form", "Set Default Dir"))
    
    
    def runButtonPressed(self):
        show_webcam_with_model(mirror=True)
    
    def inputVideoButtonPressed(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select a File", "", "", options=options)
        if fileName:  # If a filename is successfully retrieved, add picture to window
            processVideo(self, fileName)

    def setDefaultDirectoryButtonPressed(self):
        output_directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Output Directory", "", QtWidgets.QFileDialog.ShowDirsOnly)
        print(output_directory)



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
  # IMPORTANT THIS NEEDS TO HAVE THE BASE FILE LOCATION this is where the label and images directories will be created and files saved\
                            # for example C:/UserName/Desktop/test/
    base_file_name = now.strftime(
        "%Y-%m-%d_%H%M")  # create file name base for screenshot and annotation file, this could be set to the current data/time
    image_directory = output_directory + "images/"
    label_directory = output_directory + "labels/"
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)
    cv2.imwrite(image_directory + base_file_name + ".jpg", img)
    if not os.path.exists(label_directory):
        os.makedirs(label_directory)
    generate_annotation_file(label_directory + base_file_name + ".txt", detections, w, h)

app = QtWidgets.QApplication(sys.argv)
GUI = Window()

sys.exit(app.exec())
