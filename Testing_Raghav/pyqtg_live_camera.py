from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np
from models_espros_660 import Camera

def sensor_data(n_sensors, x_res, y_res):#
    return np.random.rand(n_sensors, x_res, y_res)


class Thread(QtCore.QThread):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        #global camera
    dataChanged = QtCore.pyqtSignal(np.ndarray)
    def run(self):
        import time
        then = time.time()
        for _ in range(50):
            data = camera.get_frame()#sensor_data(4, 10, 10)
            self.dataChanged.emit(data)
            QtCore.QThread.msleep(10)
        now = time.time()
        print("Frame rate: ", (50/(now-then)))

class App(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)

        self.canvas = pg.GraphicsLayoutWidget()
        self.label = QtGui.QLabel()

        lay = QtGui.QVBoxLayout(self.mainbox)
        lay.addWidget(self.canvas)
        lay.addWidget(self.label)

        self.img_items = []
        n = 1

        for i in range(n):
            for j in range(n):
                view = self.canvas.addViewBox(i, j)
                view.setAspectLocked(True)
                #view.setRange(QtCore.QRectF(0, 0, 10, 10))
                it = pg.ImageItem(None, border="w")
                view.addItem(it)
                self.img_items.append(it)
        print('AAA',len(self.img_items))
    @QtCore.pyqtSlot(np.ndarray)
    def update_data(self, data):
        #for i, v in enumerate(data):
        #    self.img_items[i].setImage(v)
        print(data.shape)
        self.img_items[0].setImage(data[:,:,0])

if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    camera = Camera(0)
    camera.initialize()

    thread = Thread(camera)
    thread.dataChanged.connect(thisapp.update_data)
    thread.start()
    thisapp.show()
    sys.exit(app.exec_())
