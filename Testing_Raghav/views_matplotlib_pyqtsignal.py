import numpy as np

#from PyQt5.QtCore import Qt, QThread, QTimer,QRectF
#from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QSlider
#from pyqtgraph import ImageView
#import pyqtgraph as pg
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib import cm
from mpl_cmaps_in_ImageItem import cmapToColormap
#from matplotlib import pyplot as plt

from utils_black import plotting

import time
class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        #self.button_frame = QPushButton('Acquire Frame', self.central_widget)
        #self.button_movie = QPushButton('Start Movie', self.central_widget)
        layout = QtWidgets.QVBoxLayout(self.central_widget)
        #self.number_images = number_images
        #static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        #layout.addWidget(static_canvas)
        #self.addToolBar(NavigationToolbar(static_canvas, self))

        #self._static_ax = static_canvas.figure.subplots()
        #t = np.linspace(0, 10, 501)
        #self._static_ax.plot(t, np.tan(t), ".")


        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(dynamic_canvas)
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                        NavigationToolbar(dynamic_canvas, self))
        self._dynamic_ax = dynamic_canvas.figure#.add_subplot(111)
        
        #self.update_timer = QtCore.QTimer()
        #self.update_timer.timeout.connect(self.update_movie)        
        
    @QtCore.pyqtSlot(list,list)
    def update_data(self, to_display,imaging_type):
        #for i, v in enumerate(data):
        #    self.img_items[i].setImage(v)
        print(to_display[2])
        pre_time = to_display[0]
        post_time = to_display[1]
        camera_index = to_display[3]
        to_display = to_display[2]
        print('update view',np.mean(to_display[0]),time.time())
        #print(data.shape)
        print(imaging_type)
        self._dynamic_ax.clear()
        #self._dynamic_ax.imshow(np.fliplr(data[:,:,1].transpose()))
        #to_display = []
        #to_display.append(data)
        #to_display.append(data[:,:,1])
        #imaging_type = ['Dist_Ampl']
        plotting(to_display,imaging_type,canvas_figure = self._dynamic_ax)
        
        #self._dynamic_ax.figure.canvas.draw()
        
                   




    

if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.resize(800,800)
    window.show()
    app.exit(app.exec_())
