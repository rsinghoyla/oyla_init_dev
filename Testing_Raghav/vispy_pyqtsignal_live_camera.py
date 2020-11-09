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
from models_espros_660 import Camera
#from models import Camera
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

        l_nbr_steps = QLabel("Nbr images ")
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
            #print(i,"------------")
            data = camera.get_frame()#sensor_data(4, 10, 10)
            self.dataChanged.emit(data)
            QThread.msleep(10)
            #print(i,"------------")
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
        #print(data.shape, "-----------------------")
        self.canvas.set_data(self.props.nbr_steps.value(),
                             self.props.combo.currentText(),data)
class Canvas(scene.SceneCanvas):

    def __init__(self,max_number_images=6):
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
        self.vbs = []
        self.images = []
        for _ in range(max_number_images):
            self.vbs.append(scene.widgets.ViewBox(border_color='red', parent=self.scene))
            self.images.append(scene.visuals.Image(parent = self.vbs[-1].scene))
            self.images[-1].cmap = 'grays'
        self.grid.add_widget(self.vbs[0],0,0)
            
        self.number_images = 1
        
        
        #self.vb2 = scene.widgets.ViewBox(border_color='green', parent=self.scene)
        #self.vb3 = scene.widgets.ViewBox(border_color='red', parent=self.scene)
        
        #self.grid.add_widget(self.vb1,0,0)
        #self.grid.add_widget(self.vb2,0,1)
        #self.grid.add_widget(self.vb3,0,2)
        
        #self.image = scene.visuals.Image(parent = self.vbs['o'].scene)
        #self.image1 = scene.visuals.Image(parent = self.vb2.scene)
        #self.image2 = scene.visuals.Image(parent = self.vb3.scene)

        
        #self.grid.add_widget(self.vb2,0,0)
        #self.grid.add_widget(self.vb,0,1)
        #self.grid.add_widget(self.vb3,0,1)

        self.freeze()

        # Add a 3D axis to keep us oriented
        #scene.visuals.XYZAxis(parent=self.view.scene)

    def update_canvas(self,number_images):
        for i in range(self.number_images):
            self.vbs[i].border_color = None
            self.grid.remove_widget(self.vbs[i])
            print(self.images[i])
            self.images[i].set_data(np.zeros((1,1)))
        #print('xxxx')    
        self.number_images = number_images
        for i in range(self.number_images):
            self.grid.add_widget(self.vbs[i],i//2,i%2)
        
    def set_data(self, n_images, cmap,frame=None):
        #self.iso.set_color(cmap)
        #cl = np.linspace(-self.radius, self.radius, n_levels + 2)[1:-1]
        #self.iso.levels = cl
        global image_count
        self.images[0].cmap = 'grays'
        if frame is not None:
            print('update view',np.mean(frame),time.time())
            #print("-------",image_count)
            # if image_count >= 0 and image_count <100:
            #     for i in range(self.number_images):
            #         self.images[i].set_data(np.rot90(frame[:,:,0]))
            # if image_count == 0:
            #     for i in range(self.number_images):
            #         self.vbs[i].camera = scene.PanZoomCamera(aspect=1)
            #         self.vbs[i].camera.set_range()
            
            # if(image_count==100):
            #     self.update_canvas(3)
            # if image_count >= 100 and image_count <200:
            #     for i in range(self.number_images):
            #         self.images[i].set_data(np.rot90(frame[:,:,0]))
            # if image_count == 100:
            #     for i in range(self.number_images):
            #         self.vbs[i].camera = scene.PanZoomCamera(aspect=1)
            #         self.vbs[i].camera.set_range()
                    
            if(image_count==0):
                self.update_canvas(4)
            if image_count >= 0:
                for i in range(self.number_images):
                    self.images[i].set_data(np.rot90(frame[:,:,i]))
            if image_count == 0:
                for i in range(self.number_images):
                    self.vbs[i].camera = scene.PanZoomCamera(aspect=1)
                    self.vbs[i].camera.set_range()
            print(image_count,self.grid)        
            
            # self.vb2.camera = scene.PanZoomCamera(aspect=1)
            # self.vb2.camera.set_range()

            # if (image_count >= 100 and image_count<200):
            #     self.images[0].set_data(np.rot90(frame[:,:,0]))
            # else:
            #     self.images[0].set_data(None)
                
            # if (image_count >=50):
            #     self.image2.set_data(np.rot90(frame[:,:,0]))
            
            
            # if(image_count==100):
            #     self.grid.add_widget(self.vbs[0],0,2)
            #     self.vbs[0].camera = scene.PanZoomCamera(aspect=1)
            #     self.vbs[0].camera.set_range()
                
            # if(image_count==50):
            #     self.vb3.camera = scene.PanZoomCamera(aspect=1)
            #     self.vb3.camera.set_range()

            # if image_count == 200:
            #     self.grid.remove_widget(self.vbs[0])
            
                
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
