import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot



class Window(QtWidgets.QWidget):

	def __init__(self): #initialize the initial window
		super().__init__()
		self.setGeometry(100,100,300,400)
		self.setWindowTitle('Fish Recognition Prototype')
		browse_button = QtWidgets.QPushButton("Browse", self)
		execute_button = QtWidgets.QPushButton("Execute", self)
		self.initButtons(browse_button,execute_button)
		v_box = QtWidgets.QVBoxLayout()
		v_box = self.addVBoxLayout(v_box,browse_button,execute_button)
		self.setLayout(v_box)
		self.show()
	
	def initButtons(self, browse_button, execute_button):
		browse_button.clicked.connect(self.open_dialog)

	def addVBoxLayout(self, v_box,browse_button, execute_button): 
		#QtWidgets.connect()
		v_box.addWidget(browse_button)
		v_box.addWidget(execute_button)
		return v_box

	@pyqtSlot()
	def open_dialog(self):
		self.openFileDialog()


	def openFileDialog(self):
		options = QtWidgets.QFileDialog.Options()
		options |= QtWidgets.QFileDialog.DontUseNativeDialog
		fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Select a File", "","All Files (*);;Image Files (*.png)", options=options)
		if fileName:
			print(fileName)
		

app = QtWidgets.QApplication(sys.argv)
GUI = Window()
sys.exit(app.exec())










