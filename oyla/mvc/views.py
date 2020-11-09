##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Creation:          20.07.2019                                                                  #
# Version:           1.0                                                                         #
# Revision history:  Initial version                                                             #
# Version : 2.0 20.10.2019
# added filtering, more controls, many more 
# Description     Some of this could be cleaner, especially all the widget stuff
# all point cloud related view classes and functions are here
##################################################################################################

from __future__ import division, print_function, absolute_import
from vispy import app, visuals
from vispy.visuals import transforms
from vispy.io import load_data_file
from vispy import scene 
from vispy.color import get_colormap, Colormap

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt, QThread, pyqtSlot
from matplotlib import cm
import numpy as np ## numpy array
import time
import cv2

import oyla.mvc.range.views as rv
import oyla.mvc.pcl.views as pcl

# Parameters that have a text field are specified here
PARAMETERS = [('Azimuth',        -360,      360,    'int', 0),
              ('Elevation',      -360,      360,    'int', -90),
              ('X_translate',        -1999,    1999,    'int', 100),#100),
              ('Y_translate',        -1999,    1999,    'int', 150),#100),
              ('Z_translate',        -19999,    19999,    'int', 750),#350),
              ('Scale',              1,   40000,    'int', 1000),#3000),
              ('FOV_angle_x',1,200,'int',90),#90),
              ('FOV_angle_y',1,200,'int',70),#70),
              ('Range_max',1,200*100+1,'double',450),
              ('Range_min',0,200*100,'double',250),
              ('Ampl_min',50,550,'double',50),
              ('X_max',-5000,5000,'double',2500),#350 # this and below is used to threshold data for a coordinate range
              ('X_min',-5000,5000,'double',-2500),#50
              ('Y_max',-5000,5000,'double',2500),#350
              ('Y_min',-5000,5000,'double',-2500),
              ('Qp_Phase',1,512,'int',1),
              ('Qp_Ampl',1,512,'int',1)]



# Parameters display name to variable name
# from Parameters list above to a dictionary which is easy to maninpulate
CONVERSION_DICT = {'Azimuth' : 'azimuth', 'Elevation' : 'elevation', 
                   'X_translate' : 'x_translate', 'Y_translate':'y_translate',
                   'Z_translate' : 'z_translate',
                    'Scale': 'scale','FOV_angle_x':'fov_angle_x','FOV_angle_y':'fov_angle_y',
                   'Range_min':'range_min','Range_max':'range_max','X_max':'x_max','X_min':'x_min',
                   'Y_max':'y_max','Y_min':'y_min','Ampl_min':'ampl_min','Qp_Phase':'qp_phase','Qp_Ampl':'qp_ampl'}
                  
                  

class Paramlist(object):
    """
    Parameters that are in drop down list are specified here; Use of Conversion_DICT
    """
    def __init__(self, parameters):
        self.parameters = parameters
        self.props = dict()
        # add the buttons and drop down ui stuff here
        self.props['bounding_box'] = False
        self.props['reflectivity_thresh'] = False
        self.props['colorize'] = False
        #self.props['colormap_type'] = 'grays'
        self.props['view_type'] = 'default_view'
        self.props['camera_type'] = 'arcball'
        self.props['transform_type'] = 'cartesian'
        # see well done
        for nameV, minV, maxV, typeV, iniV in parameters:
            nameV = CONVERSION_DICT[nameV]
            self.props[nameV] = iniV


class SetupWidget(QtWidgets.QWidget):
    """
    Setup Widget class that will creates dropdowns, textboxes for the parameters
    """
    changed_parameter_sig = QtCore.pyqtSignal(Paramlist)
    stop_loop = QtCore.pyqtSignal()
    run_loop = QtCore.pyqtSignal()
    step_forward = QtCore.pyqtSignal()
    step_backward = QtCore.pyqtSignal()
    def __init__(self, parent=None, save = False,live=True):
        
        super(SetupWidget, self).__init__(parent)

        
        # Create the parameter list from the default parameters given here
        self.param = Paramlist(PARAMETERS)
        # bounding box is a bad nomer -- its basically the box defined by the axis and the ranges for each coordinate
        # Checkbox for whether or not the bounding_box point is visible
        #self.bounding_box_chk = QtWidgets.QCheckBox(u"Show Bounding Box")
        #self.bounding_box_chk.setChecked(self.param.props['bounding_box'])
        #self.bounding_box_chk.toggled.connect(self.update_parameters)
        self.reflectivity_thresh_chk = QtWidgets.QCheckBox(u"On")
        self.reflectivity_thresh_chk.setChecked(False)
        self.reflectivity_thresh_chk.toggled.connect(self.update_parameters)
        self.colorize_chk = QtWidgets.QCheckBox(u"On")
        self.colorize_chk.setChecked(False)
        self.colorize_chk.toggled.connect(self.update_parameters)
        if save:
            self.reflectivity_thresh_chk.setEnabled(False)
            self.colorize_chk.setEnabled(False)
        # A drop-down menu for selecting which method to use for updating
        # here just for legacy
        # self.colormap_type_list = ['autumn', 'grays','viridis']
        # self.colormap_type_options = QtWidgets.QComboBox()
        # self.colormap_type_options.addItems(self.colormap_type_list)
        # self.colormap_type_options.setCurrentIndex(
        #     self.colormap_type_list.index((self.param.props['colormap_type']))
        # )
        # self.colormap_type_options.currentIndexChanged.connect(
        #     self.update_parameters
        # )
        
        # this is how to add options to drop down
        self.view_type_list = ['default_view','camera_view','bird_eye_view','other_axis_view']
        self.view_type_options = QtWidgets.QComboBox()
        self.view_type_options.addItems(self.view_type_list)
        # above the params prop will set the default method
        self.view_type_options.setCurrentIndex(
            self.view_type_list.index((self.param.props['view_type']))
        )
        self.view_type_options.currentIndexChanged.connect(
            self.update_parameters
        )
        
        
        self.transform_type_list = ['spherical', 'cartesian','cartesian_no_shift']
        self.transform_type_options = QtWidgets.QComboBox()
        self.transform_type_options.addItems(self.transform_type_list)
        self.transform_type_options.setCurrentIndex(
            self.transform_type_list.index((self.param.props['transform_type']))
        )
        self.transform_type_options.currentIndexChanged.connect(
            self.update_parameters
        )

        self.camera_type_list = ['turntable','arcball']
        self.camera_type_options = QtWidgets.QComboBox()
        self.camera_type_options.addItems(self.camera_type_list)
        self.camera_type_options.setCurrentIndex(
            self.camera_type_list.index((self.param.props['camera_type']))
        )
        self.camera_type_options.currentIndexChanged.connect(
            self.update_parameters
        )

        # Separate the different parameters into groupboxes,
        # so there's a clean visual appearance
        self.parameter_groupbox = QtWidgets.QGroupBox(u"Orientation ")
        self.conditions_groupbox = QtWidgets.QGroupBox(u"Initial Condition")
        self.display_groupbox = QtWidgets.QGroupBox(u"Display")

        self.groupbox_list = [self.parameter_groupbox,
                              self.conditions_groupbox,
                              self.display_groupbox]

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Get ready to create all the spinboxes with appropriate labels
        # https://www.computerhope.com/jargon/s/spinbox.htm
        plist = []
        self.psets = []
        # important_positions is used to separate the
        # parameters into their appropriate groupboxes
        important_positions = [0, ]
        param_boxes_layout = [QtWidgets.QGridLayout(),
                              QtWidgets.QGridLayout(),
                              QtWidgets.QGridLayout()]
        count = 0                     
        for nameV, minV, maxV, typeV, iniV in self.param.parameters:
            # Create Labels for each element
            plist.append(QtWidgets.QLabel(nameV))

            if nameV == 'scale':
                important_positions.append(len(plist) - 1)

            # Create Spinboxes based on type - doubles get a DoubleSpinBox,
            # ints get regular SpinBox.
            # Step sizes are the same for every parameter except font size.
            if typeV == 'double':
                self.psets.append(QtWidgets.QDoubleSpinBox())
                self.psets[-1].setDecimals(2)
                self.psets[-1].setSingleStep(1.0)
            elif typeV == 'int':
                self.psets.append(QtWidgets.QSpinBox())

            if save and count>7:
                self.psets[-1].setEnabled(False)
            # Set min, max, and initial values
            self.psets[-1].setKeyboardTracking(False)
            self.psets[-1].setMaximum(maxV)
            self.psets[-1].setMinimum(minV)
            self.psets[-1].setValue(iniV)
            count +=1

        # Grouping is basically happening herel quite  a bit of this is unclear including role of important_positions
        pidx = -1
        for pos in range(len(plist)):
            if pos in important_positions:
                pidx += 1
            param_boxes_layout[pidx].addWidget(plist[pos], pos + pidx, 0)
            param_boxes_layout[pidx].addWidget(self.psets[pos], pos + pidx, 1)
            self.psets[pos].valueChanged.connect(self.update_parameters)
        
        #param_boxes_layout[2].addWidget(QtWidgets.QLabel('Colormaps: '), 8, 0)
        #param_boxes_layout[2].addWidget(self.colormap_type_options, 8, 1)
        # this lays out the drop down boxes
        
        param_boxes_layout[1].addWidget(QtWidgets.QLabel('View: '), 12, 0)
        param_boxes_layout[1].addWidget(self.view_type_options, 12, 1)
        param_boxes_layout[0].addWidget(QtWidgets.QLabel('Reflectivity T'), 17, 0)
        param_boxes_layout[0].addWidget(self.reflectivity_thresh_chk, 17, 1)
        param_boxes_layout[0].addWidget(QtWidgets.QLabel('Colorize'), 18, 0)
        param_boxes_layout[0].addWidget(self.colorize_chk, 18, 1)
        param_boxes_layout[1].addWidget(QtWidgets.QLabel('Camera: '), 11, 0)
        param_boxes_layout[1].addWidget(self.camera_type_options, 11, 1)
        
        param_boxes_layout[1].addWidget(QtWidgets.QLabel('Transform: '), 8, 0)
        param_boxes_layout[1].addWidget(self.transform_type_options, 8, 1)
        
        pButton = QtWidgets.QPushButton('Pause ')
        param_boxes_layout[2].addWidget(pButton, 7, 0)
        pButton.clicked.connect(lambda:self.stop_loop.emit())
        # emit the signal that run or pause has been pressed
        rButton = QtWidgets.QPushButton('Run ')
        param_boxes_layout[2].addWidget(rButton, 7, 1)
        rButton.clicked.connect(lambda:self.run_loop.emit())
        
        sfButton = QtWidgets.QPushButton('> ')
        param_boxes_layout[2].addWidget(sfButton, 9, 0)
        sfButton.clicked.connect(lambda:self.step_forward.emit())
        # emit the signal that run or pause has been pressed
        sbButton = QtWidgets.QPushButton('< ')
        param_boxes_layout[2].addWidget(sbButton, 9, 1)
        sbButton.clicked.connect(lambda:self.step_backward.emit())
        
        if live:
            rButton.setEnabled(False)
            pButton.setEnabled(False)
        for groupbox, layout in zip(self.groupbox_list, param_boxes_layout):
            groupbox.setLayout(layout)

        for groupbox in self.groupbox_list:
            self.splitter.addWidget(groupbox)

        vbox = QtWidgets.QVBoxLayout()
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.splitter)
        hbox.addStretch(5.0)
        vbox.addLayout(hbox)
        vbox.addStretch(1.0)

        self.setLayout(vbox)

    def update_parameters(self, option):
        """When the system parameters change, get the state and emit it."""
        #self.param.props['bounding_box'] = self.bounding_box_chk.isChecked()
        self.param.props['reflectivity_thresh'] = self.reflectivity_thresh_chk.isChecked()
        self.param.props['colorize'] = self.colorize_chk.isChecked()
        self.param.props['camera_type'] = self.camera_type_list[
            self.camera_type_options.currentIndex()]
        
        # self.param.props['colormap_type'] = self.colormap_type_list[
        #     self.colormap_type_options.currentIndex()]  
        
        self.param.props['view_type'] = self.view_type_list[
            self.view_type_options.currentIndex()]
        
        self.param.props['transform_type'] = self.transform_type_list[
            self.transform_type_options.currentIndex()]
            
        keys = map(lambda x: x[0], self.param.parameters)
        for pos, nameV in enumerate(keys):
            self.param.props[CONVERSION_DICT[nameV]] = self.psets[pos].value()
        # update parameterds depends on signals like data
        self.changed_parameter_sig.emit(self.param)
        
        
class MainWindow(QtWidgets.QMainWindow):
    """
    MainWindow class initializes the Setupwidget class and teh Canvas class
    """

    def __init__(self, view_as, param=None, chips = None,rgb = False,max_number_images=3, show_colorbar = [],save = False,live=True):
        """Main Window for holding the Vispy Canvas and the parameter
        control menu.
        """
        QtWidgets.QMainWindow.__init__(self)
                
        self.resize(1000, 800)
        self.setWindowTitle('Oyla Pointcloud Visualizer')
        #self.save = savex
        self.setup_widget = SetupWidget(self,save=save,live=live)
        self.setup_widget.param = (param if param is not None else
                                       self.setup_widget.param)
        self.setup_widget.changed_parameter_sig.connect(self.update_view)
        self.stop_loop_signal = self.setup_widget.stop_loop
        self.run_loop_signal = self.setup_widget.run_loop
        self.step_forward_signal = self.setup_widget.step_forward
        self.step_backward_signal = self.setup_widget.step_backward
        self.chips = chips
        print('scc',show_colorbar)
        if view_as == 'pcl':
            self.canvas = pcl.Canvas(**self.setup_widget.param.props,setup_widget = self.setup_widget,rgb = rgb)
        elif view_as == 'range':
            self.canvas = rv.Canvas(**self.setup_widget.param.props, setup_widget = self.setup_widget, max_number_images =max_number_images,chips = chips,
                                    show_colorbar = show_colorbar)
        self.canvas.create_native()
        self.canvas.native.setParent(self)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.setup_widget)
        splitter.addWidget(self.canvas.native)

        self.setCentralWidget(splitter)

    def update_view(self, param):
        """
        Update the VisPy canvas when the parameters change.
        """

        self.canvas.set_parameters(**param.props)
                  
    #@pyqtSlot(list,list)
    def update_data(self, data,imaging_type):
        return self.canvas.set_data(data,imaging_type)
        

                  
                  
                  
                   
