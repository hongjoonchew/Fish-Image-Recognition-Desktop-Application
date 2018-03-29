from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import numpy as np

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES),3))

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")


def show_webcam_with_model(mirror=False):
    cam = cv2.VideoCapture(0)
    isWebcamOn = True
    while isWebcamOn:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        
        (h,w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
        
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
            cv2.rectangle(img, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLORS[idx], 2)
        cv2.imshow('Project Pisces Cam', img)
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            isWebcamOn = False
    cv2.destroyAllWindows()

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('Project Pisces Cam', img)
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
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
        self.runButton.setGeometry(QtCore.QRect(50, 400, 200, 40))
        self.runButton.setObjectName("runButton")
        self.runButton.setStyleSheet("background-color: #FFFFFF")
        self.runwButton = QtWidgets.QPushButton(Form)
        self.runwButton.setGeometry(QtCore.QRect(270, 400, 200, 40))
        self.runwButton.setObjectName("runwButton")
        self.runwButton.setStyleSheet("background-color: #FFFFFF")
        self.initButtons(self.runButton, self.runwButton)
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
    
    def initButtons(self, runButton, runwButton):
        runButton.clicked.connect(self.runButtonPressed)
        runwButton.clicked.connect(self.runwButtonPressed)
    
    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Project Pisces"))
        self.runButton.setText(_translate("Form", "Run Model"))
        self.runwButton.setText(_translate("Form", "Run Without Model"))
    
    
    def runButtonPressed(self):
        show_webcam_with_model(mirror=True)
    
    def runwButtonPressed(self):
        show_webcam(mirror=True)


app = QtWidgets.QApplication(sys.argv)
GUI = Window()

sys.exit(app.exec())
