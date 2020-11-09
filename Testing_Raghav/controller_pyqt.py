from PyQt5.QtWidgets import QApplication

from views_pyqt import StartWindow
from models_espros_660 import Camera as espros
from models import Camera as webcam

espros_flag = 0
if espros_flag:
    camera = espros(0)
else:
    camera = webcam(0)
camera.initialize()

app = QApplication([])
start_window = StartWindow(camera,1)

start_window.resize(360*1.5,240*1.5)
start_window.show()
app.exit(app.exec_())
