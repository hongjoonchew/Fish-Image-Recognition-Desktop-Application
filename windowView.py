from PyQt5 import QtCore, QtGui, QtWidgets
import sys


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
        self.browseButton = QtWidgets.QPushButton(Form)
        self.browseButton.setGeometry(QtCore.QRect(20, 400, 100, 40))
        self.browseButton.setObjectName("browseButton")
        self.runButton = QtWidgets.QPushButton(Form)
        self.runButton.setGeometry(QtCore.QRect(380, 400, 100, 40))
        self.runButton.setObjectName("runButton")
        self.deleteButton = QtWidgets.QPushButton(Form)
        self.deleteButton.setGeometry(QtCore.QRect(200, 400, 100, 40))
        self.runButton.setObjectName("deleteButton")
        self.initButtons(self.browseButton, self.runButton, self.deleteButton)
        self.imageLabel = QtWidgets.QLabel(Form)
        self.imageLabel.setGeometry(QtCore.QRect(20, 10, 460, 330))
        self.imageLabel.setObjectName("imageLabel")
        self.imageLabel.setStyleSheet('background-color: white')
        self.filePathLabel = QtWidgets.QLabel(Form)
        self.filePathLabel.setGeometry(20, 370, 460, 20)
        self.filePathLabel.setObjectName("filePathLabel")
        self.filePathLabel.setStyleSheet('background-color: #DDDDDD; border: 1px solid #CCCCCC')

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        self.show()

    def initButtons(self, browseButton, runButton, deleteButton):
        browseButton.clicked.connect(self.browseButtonPressed)
        runButton.clicked.connect(self.runButtonPressed)
        deleteButton.clicked.connect(self.deleteButtonPressed)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "UnderwaterROV"))
        self.browseButton.setText(_translate("Form", "Browse.."))
        self.runButton.setText(_translate("Form", "Run"))
        self.deleteButton.setText(_translate("Form", "Delete"))

    def browseButtonPressed(self):
        self.openFileDialog()

    def openFileDialog(self): #Open file explorer on browse button click
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select a File", "",
                                                            "Image Files (*.png, *.jpg)", options=options)
        if fileName: #If a filename is successfully retrieved, add picture to window
            self.filePath = fileName
            self.filePathLabel.setText(self.filePath)

    def runButtonPressed(self):
        if self.filePath:
            img = QtGui.QPixmap(self.filePath)
            self.imageLabel.setPixmap(img)
            self.imageLabel.setScaledContents(True)
            self.imageLabel.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            self.imageLabel.show()
            self.filePath = ""

        else:
            pass

    def deleteButtonPressed(self):
        self.imageLabel.clear()
        self.filePathLabel.clear()


app = QtWidgets.QApplication(sys.argv)
GUI = Window()
sys.exit(app.exec())