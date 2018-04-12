from PyQt4 import QtCore, QtGui
import sys
import cv2
import numpy as np
import os
import datetime
import os
import platform
import subprocess
import time
from objectDetectImage import identifyImage
from objectDetectImage import identifyVideo
import caffe


output_directory = "/"

CLASSES = ["background", "steelhead", "bass"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES),3))

CONFIDENCE_MIN = 0.2

#SSD MobileNet Setup
#net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

#DetectNet Setup Variables

model_file = "Chinook Model/chinook_deploy.prototxt"

pretrained_model = "Chinook Model/snapshot_iter_27216.caffemodel"


model_file_steelhead = "Steelhead Model/steelhead_deploy.prototxt"

pretrained_model_steelhead = "Steelhead Model/snapshot_iter_7040.caffemodel"

currently_selected_prototxt= model_file
currently_selected_model= pretrained_model
steelhead_boolean = False

def show_webcam_with_model(mirror=False):
    identifyWindowView(currently_selected_prototxt, currently_selected_model, 0, steelhead_boolean)


def processVideo(self, filepath):
    identifyWindowView(currently_selected_prototxt, currently_selected_model, str(filepath), steelhead_boolean)

def identifyWindowView(prototxt, model, filepath, steelhead_boolean):
    vid = cv2.VideoCapture(filepath)
    every_nth = 10
    counter = 0

    caffe.set_mode_gpu()

    net = caffe.Net(prototxt, model, caffe.TEST )



    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) 
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))

    while(True):
        # Capture video frame-by-frame
        ret, frame = vid.read()
        counter += 1
        
        if not ret:
            
            # Release the Video Device if ret is false
            vid.release()
            # Message to be displayed after releasing the device
            print "Released Video Resource"

        if counter%every_nth == 0:
             
             # Resize the captured frame to match the DetectNet model
            frame = cv2.resize(frame, (1280, 720), 0, 0)
            
            # Use the Caffe transformer to preprocess the frame
            data = transformer.preprocess('data', frame.astype('float16')/255)
            
            # Set the preprocessed frame to be the Caffe model's data layer
            net.blobs['data'].data[...] = data
            
            # Measure inference time for the feed-forward operation
            start = time.time()
            # The output of DetectNet is an array of bounding box predictions
            if(steelhead_boolean):
                bounding_boxes = net.forward()['bbox-list-class0'][0]
            else:
                bounding_boxes = net.forward()['bbox-list'][0]
            end = (time.time() - start)*1000
            
            # Convert the image from OpenCV BGR format to matplotlib RGB format for display
            
            # Create a copy of the image for drawing bounding boxes
            overlay = frame.copy()
            
            # Loop over the bounding box predictions and draw a rectangle for each bounding box
            for bbox in bounding_boxes:
                if  bbox.sum() > 0:
                     cv2.rectangle(overlay, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255, 0, 0), -1)
                    
            # Overlay the bounding box image on the original image
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Display the inference time per frame
            cv2.putText(frame,"Inference time: %dms per frame" % end,
                        (10,500), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            # Display the frame
            cv2.imshow('frame',frame)
            key = cv2.waitKey(1)
            if key ==  ord('q'):
                break
    cv2.destroyAllWindows()

def identifySteelheadVideo(self, filepath):
    identifyWindowView(currently_selected_prototxt, currently_selected_model, str(filepath), True)
    
def identifySteelheadImage(self, filepath):
    identifyImage(currently_selected_prototxt, currently_selected_model, str(filepath), True)

# This method uses the SSD model instead of DetectNet
'''def processImage(self, filepath):
    img = cv2.imread(filepath)
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
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
# This processImage method uses the DetectNet model instead, referring to another script.
def processImageDetectNet(self, filepath):
    identifyImage(currently_selected_prototxt, currently_selected_model,str(filepath), steelhead_boolean)

def processDroneVideo(self):
    identifyWindowView(currently_selected_prototxt, currently_selected_model, str("http://192.168.254.1:8090/?action=stream"), True)

def switchModels(self):
    global steelhead_boolean
    global currently_selected_model
    global currently_selected_prototxt
    global model_file_steelhead
    global model_file
    global pretrained_model_steelhead
    global pretrained_model
    if(steelhead_boolean):
        currently_selected_model = model_file
        currently_selected_prototxt = pretrained_model
        steelhead_boolean = False
    else:
        currently_selected_prototxt =  model_file_steelhead
        currently_selected_model = pretrained_model_steelhead
        steelhead_boolean = True

class Window(QtGui.QWidget):
    
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.setupUi(self)

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)

        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setMinimumSize(QtCore.QSize(500, 550))
        Form.setMaximumSize(QtCore.QSize(500, 550))
        Form.setStyleSheet("background-color: #632733")
        #Webcam Model Button
        self.runButton = QtGui.QPushButton(Form)
        self.runButton.setGeometry(QtCore.QRect(15, 400, 150, 40))
        self.runButton.setObjectName("runButton")
        self.runButton.setStyleSheet("background-color: #FFFFFF")
        # Input Video Button
        self.inputVideoButton = QtGui.QPushButton(Form)
        self.inputVideoButton.setGeometry(QtCore.QRect(175, 400, 150, 40))
        self.inputVideoButton.setObjectName("inputVideoButton")
        self.inputVideoButton.setStyleSheet("background-color: #FFFFFF")
        #Default Directory Button
        self.setDefaultDirectoryButton = QtGui.QPushButton(Form)
        self.setDefaultDirectoryButton.setGeometry(QtCore.QRect(335, 400, 150, 40))
        self.setDefaultDirectoryButton.setObjectName("setDefaultDirectoryButton")
        self.setDefaultDirectoryButton.setStyleSheet("background-color: #FFFFFF")
        #Drone Video Button
        self.droneVideoButton = QtGui.QPushButton(Form)
        self.droneVideoButton.setGeometry(QtCore.QRect(175, 450, 150, 40))
        self.droneVideoButton.setObjectName("droneVideoButton")
        self.droneVideoButton.setStyleSheet("background-color: #FFFFFF")
        self.inputImageButton = QtGui.QPushButton(Form)
        self.inputImageButton.setGeometry(QtCore.QRect(335,450,150,40))
        self.inputImageButton.setObjectName("inputImageButton")
        self.inputImageButton.setStyleSheet("background-color: #FFFFFF")
        # Steelhead Video Button
        self.switchModelButton = QtGui.QPushButton(Form)
        self.switchModelButton.setGeometry(QtCore.QRect(15, 450, 150, 40))
        self.switchModelButton.setObjectName("switchModelButton")
        self.switchModelButton.setStyleSheet("background-color: #FFFFFF")
        self.initButtons(self.runButton, self.inputVideoButton, self.setDefaultDirectoryButton, self.droneVideoButton, self.inputImageButton,  self.switchModelButton)
        self.imageLabel = QtGui.QLabel(Form)
        self.imageLabel.setGeometry(QtCore.QRect(20, 10, 460, 380))
        self.imageLabel.setObjectName("imageLabel")
        img = QtGui.QPixmap("logo.png")
        self.imageLabel.setPixmap(img)
        self.imageLabel.setScaledContents(True)
        self.imageLabel.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        self.imageLabel.show()


        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        self.show()

    def initButtons(self, runButton, inputVideoButton, setDefaultDirectoryButton, droneVideoButton, inputImageButton,  switchModelButton):
        runButton.clicked.connect(self.runButtonPressed)
        inputVideoButton.clicked.connect(self.inputVideoButtonPressed)
        setDefaultDirectoryButton.clicked.connect(self.setDefaultDirectoryButtonPressed)
        droneVideoButton.clicked.connect(self.droneButtonPressed)
        inputImageButton.clicked.connect(self.inputImageButtonPressed)
        switchModelButton.clicked.connect(self.switchModelButtonPressed)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Project Pisces"))
        self.runButton.setText(_translate("Form", "Run Model"))
        self.inputVideoButton.setText(_translate("Form", "Input Video"))
        self.setDefaultDirectoryButton.setText(_translate("Form", "Set Default Dir"))
        self.droneVideoButton.setText(_translate("Form", "Drone Input"))
        self.switchModelButton.setText(_translate("Form", "Chinook"))
        self.inputImageButton.setText(_translate("Form", "Input Image"))

    def droneButtonPressed(self): 
        processDroneVideo(self)

    def runButtonPressed(self):
        show_webcam_with_model(mirror=True)

    def inputVideoButtonPressed(self):
        options = QtGui.QFileDialog.Options()
        QtGui.QFileDialog.setStyleSheet(self, "background-color: #CCCCCC")
        options |= QtGui.QFileDialog.DontUseNativeDialog
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Select a File", "", "", options=options)
        if fileName:  # If a filename is successfully retrieved, add picture to window
            processVideo(self, fileName)

    def inputImageButtonPressed(self):
        options = QtGui.QFileDialog.Options()
        QtGui.QFileDialog.setStyleSheet(self, "background-color: #CCCCCC")
        options |= QtGui.QFileDialog.DontUseNativeDialog
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Select a File", "", "", options=options)
        if fileName:  # If a filename is successfully retrieved, add picture to window
            processImageDetectNet(self, fileName)

    def switchModelButtonPressed(self):
        global model_file 
        global pretrained_model 
        global model_file_steelhead 
        global pretrained_model_steelhead

        global currently_selected_prototxt
        global currently_selected_model
        global steelhead_boolean
        switchModels(self)
        _translate = QtCore.QCoreApplication.translate
        if(steelhead_boolean):
            self.switchModelButton.setText(_translate("Form", "Steelhead"))
        else:
            self.switchModelButton.setText(_translate("Form", "Chinook"))
    
    def setDefaultDirectoryButtonPressed(self):
        output_directory = QtGui.QFileDialog.getExistingDirectory(self, "Output Directory", "", QtGui.QFileDialog.ShowDirsOnly)
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

app = QtGui.QApplication(sys.argv)
GUI = Window()

sys.exit(app.exec_())

""""(h, w) = img.shape[:2]
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
               cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLORS[idx], 2)"""
