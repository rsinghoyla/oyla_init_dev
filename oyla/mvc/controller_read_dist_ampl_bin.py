##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Creation:          20.07.2019                                                                  #
# Version:           1.0                                                                         #
# Revision history:  Initial version
# Revision history: 2.0 20.09.2019
# Description                                                                                    #
#        script for reading (and displaying) dist and amplitude data from stored binary files    #
#        it can display in range or pcl format, but can read only dist and ampl data             #
# 
# https://stackoverflow.com/questions/41947237/pyqt-how-to-deal-with-qthread                     #
##################################################################################################
from __future__ import division, print_function, absolute_import

import sys
sys.path.append("C:/Users/Oyla1/Documents/GitHub/our_python_dev/Testing_Raghav/")
sys.path.append("C:/Users/Oyla1/Documents/GitHub/our_python_dev/")

import numpy as np
import string
import logging
import copy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt, QThread, pyqtSlot
import argparse
import shutil
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QTextEdit, QFileDialog,
                                 QSpinBox, QComboBox, QGridLayout, QVBoxLayout, QPushButton,
                                 QSplitter)
import time
from sys import argv
import datetime
import os
import json
from oyla.mvc.model_read_bin_data import DataRead # bin read

#import oyla.mvc.range.views as rv
#import oyla.mvc.pcl.views as pcl
from oyla.mvc.views import MainWindow
from oyla.mvc.utils import  write_files,  CAMERA_VERSION
from oyla.utils import read_csv_parameters, some_common_utility,   AMBIGUITY_DISTANCE_LUT, MODULATION_FREQUENCIES
logger = logging.getLogger(__name__)
    
        
class Thread(QThread):
    #####################################################################################
    # thread to read file using model_read_dist_ampl and update data in view; in its own thread
    # Different from MovieThread in many ways including that it never reads live
    # Also this thread can be paused unlike live thread
    # https://stackoverflow.com/questions/9075837/pause-and-resume-a-qthread                         #
    #####################################################################################
    def __init__(self,imaging_type='Dist_Ampl',rgb=False,save=None):
        super().__init__()
        self.imaging_type = imaging_type
        self.rgb = rgb
        self.mutex = QMutex()
        self.pauseCond = QWaitCondition()
        self.pause = False
        self.one_step = False
        self.backward = False
        self.i = 0
        self.save = save
    dataChanged = pyqtSignal(list,list,int)
    def run(self):

        global total_num_of_images
        then = time.time()
        #i = 0
        prt = 0
        while True:
            # this allows it to wait for pause to be released on play pressed again
            self.mutex.lock()
            if self.pause:
                self.pauseCond.wait(self.mutex)
            self.mutex.unlock()
            # Get data from folder; now I check if there are bin files, if not, that returns None, then I look for mat files to get data
            # Some complication here also to support rgb data for legacy.. so even if depth data stored in .bin, rgb in .mat
            # Going forward we should be using only mat_to_frame (have to TEST if rgb) is working
            data = []
            
            np_data = model.bin_to_frame(input_data_folder_name,file_indices[self.i],self.i,imaging_type=self.imaging_type)
            if np_data is not None and self.rgb:
                # legacy, bin depth, mat rgb
                _,rgb_data = model.mat_to_frame(input_data_folder_name,file_indices[self.i])
            elif np_data is None:
                # going forward
                np_data, rgb_data = model.mat_to_frame(input_data_folder_name,file_indices[self.i])
            elif not self.rgb:
                # always if rgb is not available
                rgb_data = np.zeros((0,0,3))

            ## j is image no. ##  0 is image type ## 0 is transform_type; hacks to keep consistent with live data stream
            
            data.append(np_data)
            # Here we are sticking to the format in live controller
            _res = [0, 1, data, 1,rgb_data]
            self.dataChanged.emit(_res,[self.imaging_type],self.i)
            QThread.msleep(500) #large number else its very fast

            if not self.backward:
                self.i += 1
            else:
                self.i -= 1
            if self.i>= total_num_of_images:
                if self.save is None:
                    self.i = 0  # cycling back
                else:
                    break
            if self.i<=0:
                if self.save is None:
                    self.i = total_num_of_images-1  # cycling back
                else:
                    break
            if self.one_step:
                self.paused()
            # needed for pause and resume
            QApplication.processEvents() 

        now = time.time()
    def paused(self):
        self.mutex.lock()
        self.pause = True
        self.one_step = False
        self.backward = False
        self.mutex.unlock()
        
    def resumed(self):
        self.mutex.lock()
        self.pause = False
        self.one_step = False
        self.backward = False
        self.mutex.unlock()
        self.pauseCond.wakeAll()
    def step_forward(self):
        if self.pause:
            self.mutex.lock()
            self.pause = False
            self.one_step = True
            self.backward = False
            self.mutex.unlock()
            self.pauseCond.wakeAll()
    def step_backward(self):
        if self.pause:
            self.mutex.lock()
            self.pause = False
            self.one_step = True
            self.backward = True
            self.mutex.unlock()
            self.pauseCond.wakeAll()
# def uncaught_exceptions(ex_type, ex_value, ex_traceback):
#     '''
#     Not sure what this does
#     '''
#     lines = traceback.format_exception(ex_type, ex_value, ex_traceback)
#     msg = ''.join(lines)
#     logger.error('Uncaught Exception\n%s', msg)

@QtCore.pyqtSlot(list,list,int)
def update_data(data,imaging_type,frame_number):
    global window
    global filters_on
    global save
    global save_path
    prt = time.time()
    filtered_phase, filtered_ampl,filtered_rgb = window.update_data(data,imaging_type)
    crt = time.time()
    print("current time",crt-prt)
    # if filters_on we store the filtered data 
    if save is not None and filters_on:
        data_f = copy.deepcopy(data)
        tmp = np.stack((filtered_phase,filtered_ampl),axis=2)
        _tmp = []
        for i,_a in enumerate(data_f[2][0]):
            if i == 0:
                _a = tmp
            _tmp.append(_a)
        data_f[2][0] = np.asarray(_tmp)

        if filtered_rgb is not None:
            data_f[4] = np.resize(data_f[4],filtered_rgb.shape)
            data_f[4] = filtered_rgb
        stored_data['filtered'][frame_number] = data_f
        key = 'filtered'
        write_files(stored_data[key][frame_number],key,save_path,fno=frame_number)
        del stored_data['filtered'][frame_number]
        
if __name__ == '__main__':
    #sys.excepthook = uncaught_exceptions
    #logging.basicConfig(level=logging.INFO)
    #logging.getLogger().setLevel(logging.INFO)
    appQt = QtWidgets.QApplication([])

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_idx_file",type = str,
                        default="../Binary_to_Pointcloud_vispy/BinaryDataDistanceAmplitude/imageDistance.idx")
    parser.add_argument("--view_as",type = str, default = "range")
    parser.add_argument("--parameters_csv",type = str, default = None)
    parser.add_argument("--save",type = str, default = None)
    parser.add_argument("--rgb",type = bool, default = True)
    parser.add_argument("--filters_csv",type = str, default = None)
    parser.add_argument("--camera_version",type = str, default = None)
    parser.add_argument("--bbox",type=str, default = None, nargs='*')
    parser.add_argument("--txt_file",type=str,default = None)
    #parser.add_argument("--temporal_filtering", action="store_false")
    args = parser.parse_args()

    # not only idx files that lists the frames, also read parameters.csv
    input_data_folder_name =  "/".join(args.input_idx_file.split("/")[:-1])
    if args.parameters_csv is None:
        csv_file = input_data_folder_name+'/parameters.csv'
    else:
        csv_file = args.parameters_csv
    parameters = read_csv_parameters(csv_file)
    num_epochs = int(parameters['param']['number_epochs'][0])
    if 'camera_version' not in parameters['param']:
        assert args.camera_version is not None, "no camera version in parameters file and not specified on command line"
        camera_version = args.camera_version
    else:
        camera_version = parameters['param']['camera_version'][0]
        if 'v2' in camera_version and args.camera_version is not None and 'v3' in args.camera_version:
            camera_version = args.camera_version
    assert camera_version in CAMERA_VERSION, "Not supported camera version"
    
    filters_on = args.filters_csv is not None
    save = args.save
    assert int(parameters['param']['number_epochs'][0]) ==1, "ONLY ONE EPOCH LOADING SUPPORTED"

    # we are reading some parameters from the csv file
    _dim = list(map(int,parameters['adaptive_cmd']['setROI'][0].split(' ')))
    width = _dim[1]-_dim[0]+1
    height = _dim[3]-_dim[2]+1
    if parameters['param']['enableVerticalBinning'][0] == '1':
        height = height // 2
    if parameters['param']['enableHorizontalBinning'][0] == '1':
        width = width //2
    if len(parameters['param']['chip_id'][0].split(',')) == 2:
        width = width *2

    chips = parameters['param']['chip_id'][0].split(',')
    imaging_type = parameters['param']['imaging_type'][0].split('+')
    imaging_mode = parameters['param']['imaging_mode'][0]
    if imaging_mode == 'HDR':
        imaging_type = ['HDR'] #a hack for now

    assert imaging_type[0] == 'Dist_Ampl' or imaging_type[0] == 'DCS' or imaging_type[0] =='HDR', "Support only Dist_Ampl and HDR and DCS at this point"
    
    # the reason this is fine because in Thread a list will be made out of it; update_data requires a list
    imaging_type = imaging_type[0]

    # consistent with controller
    ambiguity_distance, range_max, range_min, saturation_flag, adc_flag, mod_freq,ampl_min,reflectivity_thresh = some_common_utility(parameters,0)
    #range_max = 4500
    #range_min = 500
    #ampl_min = 100
    print(ampl_min)
    print('range_max',range_max,range_min)
    # read filter_params
    filter_params = {}
    if args.filters_csv is not None and (imaging_type == 'Dist_Ampl' or imaging_type == 'HDR'):
        filter_parameters = read_csv_parameters(args.filters_csv)
        for k in filter_parameters['filter_cmd'].keys():
            #print(parameters['filter_cmd'][k][self.epoch_number])
            try:
                filter_params[k] = filter_parameters['filter_cmd'][k][0]
                #parameters['filter_cmd'][k][0] = filter_parameters['filter_cmd'][k][0]
            except ValueError:
                pass
        print('FILTERS',filter_params)

    # read file indices from idx file
    total_num_of_images = sum(1 for line in open(args.input_idx_file, "r"))
    file_indices  = open(args.input_idx_file).read().splitlines()
    print(imaging_type,width,height,mod_freq)

    # model for reading data from file
    model = DataRead(width, height)
    window = MainWindow(args.view_as, rgb = args.rgb, show_colorbar=[0,1],save=save,live=False)
    
    # create windows for pcl/range and update canvas with parameters
    if args.view_as == 'pcl':
        assert imaging_type == 'Dist_Ampl' or imaging_type == 'HDR',"pcl only Dist_Ampl, HDR"
        #win = pcl.MainWindow(rgb = args.rgb)
        window.canvas.update_canvas(number_chips = len(chips),
                                    ambiguity_distance = ambiguity_distance,
                                    range_max = range_max, range_min = range_min, 
                                    saturation_flag = saturation_flag, adc_flag = adc_flag, mod_freq = mod_freq, filter_params = filter_params,rgb = args.rgb,
                                    ampl_min = ampl_min, reflectivity_thresh = reflectivity_thresh, camera_version = camera_version, bbox = args.bbox, txt_file=args.txt_file,
                                    total_num_of_images = total_num_of_images)
    else:
        #win = rv.MainWindow(show_colorbar=[0,1])
        window.resize(1300,700)
        if imaging_type == 'Dist_Ampl':
            number_images = 2
        elif imaging_type == 'HDR':
            number_images = 4
        elif imaging_type == 'DCS':
            number_images = 4
        if args.rgb:
            number_images += 1
        window.canvas.update_canvas(number_images = number_images, number_chips = len(chips),
                                    ambiguity_distance = ambiguity_distance,
                                    range_max = range_max, range_min = range_min, 
                                    saturation_flag = saturation_flag, adc_flag = adc_flag, mod_freq = mod_freq, filter_params = filter_params,rgb = args.rgb,
                                    ampl_min = ampl_min, reflectivity_thresh = reflectivity_thresh,camera_version = camera_version, total_num_of_images = total_num_of_images)
        
    # save  and filter_csv (implies filtereing) should be there to save data
    if save is not None and filters_on:
        stored_data = {}
        stored_data['filtered'] = {}
        save_path = input_data_folder_name#args.save
        t = datetime.datetime.now()    
        #save_path = os.path.join(save_path,'data_{:%B_%d_%H_%M_%S}'.format(t))
        save_path = save_path+'/filtered_data_{:%B_%d_%H_%M_%S}'.format(t)+'_'+save
        if not os.path.isdir(save_path): os.mkdir(save_path)
        #with open(save_path+'/filter_params.json','w') as fp:
        #    json.dump(filter_params,fp)
        #if not os.path.isdir(save_path): os.mkdir(save_path)
        shutil.copy(csv_file,save_path+'/parameters.csv')
        shutil.copy(args.filters_csv,save_path+'/filter_params.csv')
        
        #
            
    # this is thread that is going to read data files and emit signal whenever file is read
    thread = Thread(imaging_type,args.rgb,save)
    #thread.dataChanged.connect(win.update_data)
    thread.dataChanged.connect(update_data)

    # this was added to ensure that thread can pause and resume
    #if args.view_as == 'pcl':
    window.stop_loop_signal.connect(thread.paused)
    window.run_loop_signal.connect(thread.resumed)
    window.step_forward_signal.connect(thread.step_forward)
    window.step_backward_signal.connect(thread.step_backward)
    thread.start()
    
    
    window.show()
    if sys.flags.interactive == 0:
        appQt.exec()


    #for c in stored_data.keys():
    #    write_files(stored_data[c],c,save_path)
