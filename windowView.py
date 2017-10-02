import sys
from PyQt5 import QtWidgets

def window():
	app = QtWidgets.QApplication(sys.argv)
	w = QtWidgets.QWidget()
	boxLayout = QtWidgets.QGroupBox()
	
	w.setWindowTitle('Fish Recognition Prototype')
	w.setGeometry(100,100,300,400)
	w.show()
	sys.exit(app.exec())

window()