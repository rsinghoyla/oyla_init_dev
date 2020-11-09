# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

"""
This example demonstrates isocurve for triangular mesh with vertex data and a
 qt interface.
"""

import sys
import numpy as np
import time
from vispy import scene

from vispy.geometry.generation import create_sphere
from vispy.color.colormap import get_colormaps
#from models_espros_660 import Camera
from models import Camera
try:
    from sip import setapi
    setapi("QVariant", 2)
    setapi("QString", 2)
except ImportError:
    pass

try:
    from PyQt4.QtCore import pyqtSignal, Qt
    from PyQt4.QtGui import (QApplication, QMainWindow, QWidget, QLabel,
                             QSpinBox, QComboBox, QGridLayout, QVBoxLayout,
                             QSplitter)
except Exception:
    from PyQt5.QtCore import pyqtSignal, Qt, QThread, pyqtSlot
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                                 QSpinBox, QComboBox, QGridLayout, QVBoxLayout,
                                 QSplitter)


class ObjectWidget(QWidget):
    """
    Widget for editing OBJECT parameters
    """
    signal_object_changed = pyqtSignal(name='objectChanged')

    def __init__(self, parent=None):
        super(ObjectWidget, self).__init__(parent)

        l_nbr_steps = QLabel("Nbr Steps ")
        self.nbr_steps = QSpinBox()
        self.nbr_steps.setMinimum(3)
        self.nbr_steps.setMaximum(100)
        self.nbr_steps.setValue(6)
        self.nbr_steps.valueChanged.connect(self.update_param)

        l_cmap = QLabel("Cmap ")
        self.cmap = sorted(get_colormaps().keys())
        self.combo = QComboBox(self)
        self.combo.addItems(self.cmap)
        self.combo.currentIndexChanged.connect(self.update_param)

        gbox = QGridLayout()
        gbox.addWidget(l_cmap, 0, 0)
        gbox.addWidget(self.combo, 0, 1)
        gbox.addWidget(l_nbr_steps, 1, 0)
        gbox.addWidget(self.nbr_steps, 1, 1)

        vbox = QVBoxLayout()
        vbox.addLayout(gbox)
        vbox.addStretch(1.0)

        self.setLayout(vbox)

    def update_param(self, option):
        self.signal_object_changed.emit()

class Thread(QThread):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        #global camera
    dataChanged = pyqtSignal(np.ndarray)
    def run(self):
        import time
        then = time.time()
        for i in range(500):
            print(i,"------------")
            data = camera.get_frame()#sensor_data(4, 10, 10)
            self.dataChanged.emit(data)
            QThread.msleep(10)
            print(i,"------------")
        now = time.time()
        print("------------------------Frame rate: ", (50/(now-then)))
        
class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)

        self.resize(700, 500)
        self.setWindowTitle('vispy example ...')

        splitter = QSplitter(Qt.Horizontal)

        self.canvas = Canvas()
        self.canvas.create_native()
        self.canvas.native.setParent(self)

        self.props = ObjectWidget()
        splitter.addWidget(self.props)
        splitter.addWidget(self.canvas.native)

        self.setCentralWidget(splitter)
        self.props.signal_object_changed.connect(self.update_view)
        self.update_view()

    def update_view(self):
        # banded, nbr_steps, cmap
        self.canvas.set_data(self.props.nbr_steps.value(),
                             self.props.combo.currentText())

    @pyqtSlot(np.ndarray)
    def update_data(self, data):
        #for i, v in enumerate(data):
        #    self.img_items[i].setImage(v)
        print(data.shape, "-----------------------")
        self.canvas.set_data(self.props.nbr_steps.value(),
                             self.props.combo.currentText(),data)
class Canvas(scene.SceneCanvas):

	def __init__(self):
		scene.SceneCanvas.__init__(self, keys=None)
		self.size = 800, 600
		self.unfreeze()
		#self.view = self.central_widget.add_view()
		self.grid = self.central_widget.add_grid(margin=10)
		# self.radius = 2.0
		# self.view.camera = 'turntable'
		# mesh = create_sphere(20, 20, radius=self.radius)
		# vertices = mesh.get_vertices()
		# tris = mesh.get_faces()

		# cl = np.linspace(-self.radius, self.radius, 6 + 2)[1:-1]

		# self.iso = scene.visuals.Isoline(vertices=vertices, tris=tris,
		#                                  data=vertices[:, 2],
		#                                  levels=cl, color_lev='autumn',
		#                                  parent=self.view.scene)

		self.vb1 = scene.widgets.ViewBox(border_color='blue', parent=self.scene)
		self.vb2 = scene.widgets.ViewBox(border_color='green', parent=self.scene)
		self.vb3 = scene.widgets.ViewBox(border_color='red', parent=self.scene)
		
		self.grid.add_widget(self.vb1,0,0)
		self.grid.add_widget(self.vb2,0,1)
		self.grid.add_widget(self.vb3,0,2)
		self.image = scene.visuals.Image(parent = self.vb1.scene)
		self.image1 = scene.visuals.Image(parent = self.vb3.scene)
		#self.image2 = scene.visuals.Image(parent = self.vb3.scene)
		self.freeze()

		# Add a 3D axis to keep us oriented
		#scene.visuals.XYZAxis(parent=self.view.scene)

	def set_data(self, n_levels, cmap,frame=None):
		#self.iso.set_color(cmap)
		#cl = np.linspace(-self.radius, self.radius, n_levels + 2)[1:-1]
		#self.iso.levels = cl
		global image_count
		if frame is not None:
			print('update view',np.mean(frame),time.time())
			print("-------",image_count)
			self.image.set_data(np.rot90(frame[:,:,0]))
			self.image1.set_data(np.rot90(frame[:,:,0]))
			if(image_count==100):
				self.vb2.border_color = None
				self.vb3.border_color = None
				self.grid.remove_widget(self.vb2)
				self.grid.remove_widget(self.vb3)
				
			
			if(image_count==300):
				self.vb3.border_color = 'red'
				self.grid.add_widget(self.vb3,0,1)	
			#self.image1.set_data(np.rot90(frame[:,:,0]))
			self.vb1.camera = scene.PanZoomCamera(aspect=1)
			self.vb1.camera.set_range()
			
			self.vb3.camera = scene.PanZoomCamera(aspect=1)
			self.vb3.camera.set_range()
			print(self.grid)
			self.update()#pass
			image_count+=1


image_count = 0
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    appQt = QApplication(sys.argv)
    win = MainWindow()
    camera = Camera(0)
    camera.initialize()

    thread = Thread(camera)
    thread.dataChanged.connect(win.update_data)
    thread.start()
    win.show()
    appQt.exec_()
