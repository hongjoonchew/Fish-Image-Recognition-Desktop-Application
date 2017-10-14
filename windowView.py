import sys
from PyQt5 import QtWidgets


class Window(QtWidgets.QWidget):

	def __init__(self):
		super().__init__()
		self.setGeometry(100,100,300,400)
		self.setWindowTitle('Fish Recognition Prototype')
		v_box = QtWidgets.QVBoxLayout()
		v_box = self.addVBoxLayout(v_box);
		self.setLayout(v_box)
		self.show()

	def addVBoxLayout(self, v_box):
		execute_button = QtWidgets.QPushButton("Execute");
		v_box.addWidget(execute_button)
		return v_box



		

app = QtWidgets.QApplication(sys.argv)
GUI = Window()
sys.exit(app.exec())










