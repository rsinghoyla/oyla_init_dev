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
from vispy.color import get_colormap, Colormap
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

from utils_black import convert_matrix_image, getComparedHDRAmplitudeImage_vectorized

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


        
class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)

        #self.resize(700, 500)
        self.setWindowTitle('vispy example ...')

        #splitter = QSplitter(Qt.Horizontal)

        self.canvas = Canvas()
        self.canvas.create_native()
        self.canvas.native.setParent(self)

        self.props = ObjectWidget()
        #splitter.addWidget(self.props)
        #splitter.addWidget(self.canvas.native)

        #self.setCentralWidget(splitter)
        #self.props.signal_object_changed.connect(self.update_view)
        self.update_view()

    def update_view(self):
        # banded, nbr_steps, cmap
        self.canvas.set_data(self.props.nbr_steps.value(),
                             self.props.combo.currentText())

    @pyqtSlot(list,list)
    def update_data(self, data,imaging_type):
        #for i, v in enumerate(data):
        #    self.img_items[i].setImage(v)
        #print(data.shape, "-----------------------")
        self.canvas.set_data(self.props.nbr_steps.value(),
                             self.props.combo.currentText(),data,imaging_type)
class Canvas(scene.SceneCanvas):

    def __init__(self,max_number_images=6):
        scene.SceneCanvas.__init__(self, keys=None)
        self.size = 1000, 800
        self.unfreeze()
        #self.view = self.central_widget.add_view()
        self.grid = self.central_widget.add_grid(margin=1)
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
        self.colorbars = []
        #colormap = Colormap(['r', 'g', 'b'])

        #colormap[np.linspace(0., 1., 100)]
        for image_number in range(max_number_images):
            self.colorbars.append(scene.ColorBarWidget(label="", 
                                                       cmap='grays', orientation="left", border_width=1,
                                                       border_color = "#212121",label_color="white"))
            if image_number%2==0:
                self.vbs.append(scene.widgets.ViewBox(border_color='red', parent=self.scene))
            else:
                _widget = scene.widgets.ViewBox(border_color=None, parent=self.scene)
            
                _grid = _widget.add_grid()
                _grid.add_widget(self.colorbars[-1], col = 25)
                self.vbs.append(_widget)

            
            self.images.append(scene.visuals.Image(parent = self.vbs[-1].scene))
            #self.images[-1].cmap = 'viridis'

        self.grid.add_widget(self.vbs[0],0,0)
            
        self.number_images = 1
        self.updated_canvas = True
        self.got_image = False
        self.number_cameras = 1
        
        self.ambiguity_distance = None
        self.range_max = None
        self.range_min = None
        self.saturation_flag = False
        self.adc_flag = False
        
        self.AMPLITUDE_SATURATION = 65400
        self.ADC = 65500
        self.LOW_AMPLITUDE = 65300
        self.MAX_PHASE = 30000.0
        self.GRAYSCALE_SATURATION = 4095
        
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

    def update_canvas(self,number_images, number_cameras = 1, ambiguity_distance = None,
                      range_max = 9 ,range_min = 0.1, saturation_flag = None ,
                      adc_flag = None):
        for i in range(self.number_images):
            self.vbs[i].border_color = None
            self.vbs[i].camera.reset()
            self.grid.remove_widget(self.vbs[i])
            self.images[i].set_data(None)
            #self.colorbars[i].cmap = 'gray'
        
        self.number_images = number_images
        self.number_cameras = number_cameras
        
        for i in range(self.number_images):
            self.grid.add_widget(self.vbs[i],i//2,i%2)
        self.updated_canvas = True*np.ones(self.number_images)
        self.got_image = False*np.ones(self.number_images)
        
        self.ambiguity_distance = ambiguity_distance
        self.range_max = range_max
        self.range_min = range_min
        self.saturation_flag = saturation_flag
        self.adc_flag = adc_flag
        

    def rescale_image(self,img):
        if self.ambiguity_distance:
            img = img/(self.MAX_PHASE)*self.ambiguity_distance
            #print("sum,max after scaling to ambiguity distance",ambiguity_distance,np.sum(img),np.max(img))
        if self.range_max:
            img[np.where(img>self.range_max)] = 0
        if self.range_min:
            img[np.where(img<self.range_min)] = 0
            #print("sum,max after range ",range_max, range_min ,np.sum(img),np.max(img))
        return img
    
    def set_data(self, n_images, cmap,to_display=None,imaging_type = []):
        #self.iso.set_color(cmap)
        #cl = np.linspace(-self.radius, self.radius, n_levels + 2)[1:-1]
        #self.iso.levels = cl
        global image_count
        #self.images[0].cmap = 'grays'
        
        if to_display is not None:
            #print(to_display[2])
            pre_time = to_display[0]
            post_time = to_display[1]
            camera_index = to_display[3]
            to_display = to_display[2]
            print('update view',np.mean(to_display[0]),time.time(),to_display[0].shape,self.vbs[0].camera.get_state())
            #to_display = frame
            #if(image_count==200):
            #    self.update_canvas(2)
            #if image_count >= 200:
            #for i in range(self.number_images):
            #    self.images[i].set_data(np.rot90(frame[:,:,i]))
            print(camera_index)
            cnt  = 0
            for ind in range(len(imaging_type)):
        
                if imaging_type[ind] == 'Gray':

                    grayscale_saturated_indices = np.where(np.squeeze(to_display[ind]>=self.GRAYSCALE_SATURATION))
                    print("Grayscale Saturated indices =", np.count_nonzero(grayscale_saturated_indices))
                    img = np.squeeze(to_display[ind])
                    img = convert_matrix_image(img,cmap= 'gray', clim_min=2048, clim_max=4095,
                                         saturation_indices = grayscale_saturated_indices)
                    cnt = self.plot(img,cnt,camera_index,'grays')
                elif imaging_type[ind] == 'Dist_Ampl':
            
                    #####to scale and threshold#######################################################################
                    #print("sum,max before ", np.sum(to_display[ind]))

                    amplitude_saturated_indices = np.where(np.squeeze(to_display[ind][:,:,1])>=self.AMPLITUDE_SATURATION)
                    print("Amplitude Saturated indices =", np.count_nonzero(amplitude_saturated_indices))

                    if self.saturation_flag:
                        to_display[ind][:,:,0][np.where(to_display[ind][:,:,1]==self.AMPLITUDE_SATURATION)] = 0
                    if self.adc_flag:
                        to_display[ind][:,:,0][np.where(to_display[ind][:,:,1]==self.ADC)] = 0
                        
                    #print("sum,max after saturation and adc ", saturation_flag, adc_flag, np.sum(to_display[ind]),np.max(to_display[ind]))
                    #print(amplitude_min)

                    ##            if amplitude_min:
                    ##                to_display[ind][:,:,0][np.where(to_display[ind][:,:,1]<=amplitude_min)] = 0
                    ##                to_display[ind][:,:,1][np.where(to_display[ind][:,:,1]<=amplitude_min)] = 0

                    low_amplitude_indices = np.where(np.squeeze(to_display[ind][:,:,1])==self.LOW_AMPLITUDE)
                    print("Low Amplitude Indices =", np.count_nonzero(low_amplitude_indices))

                    ##            to_display[ind][:,:,0][np.where(to_display[ind][:,:,1]==65300)] = 0
                    ##            to_display[ind][:,:,1][np.where(to_display[ind][:,:,1]==65300)] = 0

                    #print("sum,max after LSB ", amplitude_min, np.sum(to_display[ind]))

                    img = np.squeeze(to_display[ind])[:,:,0]
                    img = self.rescale_image(img)
                    img = convert_matrix_image(img,cmap= 'jet_r', clim_min=self.range_min, clim_max=self.range_max,
                                               saturation_indices = amplitude_saturated_indices,
                                               no_data_indices = low_amplitude_indices)
                    cnt = self.plot(img,cnt,camera_index,'jet_r')

                    img = np.squeeze(to_display[ind])[:,:,1]
                    img = convert_matrix_image(img,cmap= 'jet', clim_min=1, clim_max=2000,
                                               saturation_indices = amplitude_saturated_indices,
                                               no_data_indices = low_amplitude_indices)
                    cnt = self.plot(img,cnt,camera_index,'jet')

                elif imaging_type[ind] == 'Dist':

                    img = np.squeeze(to_display[ind])
                    img = self.rescale_image(img)
                    distance_saturated_indices = np.where(np.squeeze(to_display[ind])==self.AMPLITUDE_SATURATION)
                    img = np.squeeze(to_display[ind])
                    img = convert_matrix_image(img,cmap= 'jet_r', clim_min=0.5, clim_max=5,
                                         saturation_indices = distance_saturated_indices)
                    cnt = self.plot(img,cnt,camera_index,'jet_r')

                elif imaging_type[ind] == 'Ampl':
                    to_display[ind][np.where(to_display[ind]==self.low_amplitude)] = 0
                    img = np.squeeze(to_display[ind])
                    distance_saturated_indices = np.where(np.squeeze(to_display[ind])==self.AMPLITUDE_SATURATION)
                    img = convert_matrix_image(img,cmap= 'gnuplot_r', clim_min=1, clim_max=2000,
                                            saturation_indices = distance_saturated_indices)
                    cnt = self.plot(img,cnt,camera_index,'gnuplot_r')
                    
                elif imaging_type[ind] == 'DCS' :
            
                    img = np.squeeze(to_display[ind])[:,:,0]
                    median_scale =  np.median(img)
                    img = convert_matrix_image(img,cmap= 'viridis', 
                                               clim_min=median_scale-500, clim_max=median_scale+500)
                    cnt = self.plot(img,cnt,camera_index,'viridis')
                    img = np.squeeze(to_display[ind])[:,:,1]
                    img = convert_matrix_image(img,cmap= 'viridis', 
                                               clim_min=median_scale-500, clim_max=median_scale+500)
                    cnt = self.plot(img,cnt,camera_index,'viridis')
                    img = np.squeeze(to_display[ind])[:,:,2]
                    img = convert_matrix_image(img,cmap= 'viridis', 
                                               clim_min=median_scale-500, clim_max=median_scale+500)
                    cnt = self.plot(img,cnt,camera_index,'viridis')
                    img = np.squeeze(to_display[ind])[:,:,0] + np.squeeze(to_display[ind])[:,:,2]
                    img = convert_matrix_image(img,cmap= 'viridis', 
                                               clim_min=2*median_scale-200, clim_max=2*median_scale+200)
                    cnt = self.plot(img,cnt,camera_index,'viridis')
                    
                elif imaging_type[ind] == 'HDR':
            
                    img = np.squeeze(to_display[ind])[:,:,0]
                    amplitude_saturated_indices = np.where(np.squeeze(to_display[ind][:,:,0])>=self.AMPLITUDE_SATURATION)
                    ## if manipulating img directly (as is done on the basis of range min/max below) saturation indices must be identified before zero distance indices
                    img = self.rescale_image(img)
                    zero_distance_indices = np.where(np.squeeze(img)==0)
                    img = convert_matrix_image(img,cmap= 'jet_r',  clim_min=self.range_min, clim_max=self.range_max,
                                               saturation_indices = amplitude_saturated_indices,
                                               no_data_indices = zero_distance_indices)
                    cnt = self.plot(img,cnt,camera_index,'jet_r')

                    img = np.squeeze(to_display[ind])[:,:,1]
                    amplitude_saturated_indices = np.where(np.squeeze(to_display[ind][:,:,1])>=self.AMPLITUDE_SATURATION)
                    img = self.rescale_image(img)
                    zero_distance_indices = np.where(np.squeeze(img)==0)
                    img = convert_matrix_image(img,cmap= 'jet_r',  clim_min=self.range_min, clim_max=self.range_max,
                                               saturation_indices = amplitude_saturated_indices,
                                               no_data_indices = zero_distance_indices)
                    cnt = self.plot(img,cnt,camera_index,'jet_r')

                    img = np.squeeze(to_display[ind])[:,:,2]
                    amplitude_saturated_indices = np.where(np.squeeze(to_display[ind][:,:,2])>=self.AMPLITUDE_SATURATION)
                    img = self.rescale_image(img)
                    zero_distance_indices = np.where(np.squeeze(img)==0)
                    img = convert_matrix_image(img,cmap= 'jet_r',  clim_min=self.range_min, clim_max=self.range_max,
                                               saturation_indices = amplitude_saturated_indices,
                                               no_data_indices = zero_distance_indices)
                    cnt = self.plot(img,cnt,camera_index,'jet_r')

                    img = np.squeeze(to_display[ind])[:,:,3]
                    ampl = np.squeeze(to_display[ind])[:,:,4]
                    ampl,img = getComparedHDRAmplitudeImage_vectorized(ampl,img)
                    #print(np.all(ampl0==ampl2),np.all(img0==img2))
                    amplitude_saturated_indices = np.where(np.squeeze(ampl>=self.AMPLITUDE_SATURATION))
                    img = self.rescale_image(img)
                    zero_distance_indices = np.where(np.squeeze(img)==0)
                    img = convert_matrix_image(img,cmap= 'jet_r',  clim_min=self.range_min, clim_max=self.range_max,
                                               saturation_indices = amplitude_saturated_indices,
                                               no_data_indices = zero_distance_indices)
                    cnt = self.plot(img,cnt,camera_index,'jet_r')

                    # tmp = np.zeros((2*img.shape[0]+100,img.shape[1]))
                    # tmp[:img.shape[0],:] = img
                    # img = np.squeeze(to_display[ind])[:,:,2]
                    # tmp[img.shape[0]+100:,:] = img
                    # plot_subplot(1,1,1,tmp)

                elif imaging_type[ind] == 'DCS_2':

                    img = np.squeeze(to_display[ind])[:,:,0]
                    cnt = self.plot(img,cnt,camera_index,'viridis')
                    img = np.squeeze(to_display[ind])[:,:,1]
                    cnt = self.plot(img,cnt,camera_index,'viridis')
                    
            
            for i in range(self.number_images):
                if self.updated_canvas[i] and self.got_image[i]:
                    self.vbs[i].camera = scene.PanZoomCamera(aspect=1)
                    #self.vbs[i].camera.zoom = 2.0
                    self.vbs[i].camera.set_range()
                    self.updated_canvas[i] = False
                
            
                
            self.update()#pass
            image_count+=1

    def plot(self, img, cnt, camera_index,cmap='gray'):
        if camera_index ==0:
            self.images[cnt+self.number_cameras*camera_index].set_data(np.rot90(np.flipud(img)))
        else:
            self.images[cnt+self.number_cameras*camera_index].set_data(np.flipud(np.rot90(np.flipud(img))))
        self.images[cnt+self.number_cameras*camera_index].cmap = cmap
        self.got_image[cnt+self.number_cameras*camera_index] = True
        
        _n = (cnt+self.number_cameras*camera_index)
        print(cnt,cnt+self.number_cameras*camera_index)
        self.colorbars[_n].cmap = cmap#self.images[cnt+self.number_cameras*camera_index].cmap
        print(self.colorbars[cnt+self.number_cameras*camera_index].cmap.colors,cmap)
        self.colorbars[_n].clim  = (0.1, 9.0)
            
            
        return cnt+1
    
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
