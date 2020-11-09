##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Creation:          20.04.2019                                                                  #
# Version:           1.0                                                                         #
# Revision history:  1.1 Initial version
# One major revision (20.07.2019) is frame_number is being passed to epc_image getXXX().
# Also its assumed that epc_image will return data, along with serverIP (chip), imaging_type and chip
# Cleaning this up in major way (20.10.2019)
# introduced read_bin_file, file_type which will be either unpack or np.fromfile
# cleaned read_*_files, so as to call read_bin_file and also use image_type for dcs, hdr;
# no image_type for dist_ampl as both dist and ampl have the same size
# mat file can now read the data
# 
##################################################################################################
from __future__ import division, print_function, absolute_import
import sys
import numpy as np
import string
import logging
import struct  ## library for unpacking of bin file
import numpy as np
import time
import scipy.io
import os
class DataRead():
    """
    Binary_Data_Read class reads the data from binary file and apply unpacking to get range image.
    Then convert the range image into Pointcloud using transformation method.
    """
    def __init__(self,num_cols=320,num_rows=240):
        print("DataRead class initialized ")
    
        """ read_binary_files function to convert binary data to range image by unpacking binary value
        Input --> 
        filename = input binary data file name, outfilename = file name to write the range image, 
        image_types = format of binary data either amplitude or original or filtered data
        
        Output -->
        return range image in the form of numpy array of shape (240,320)
        """
        self.num_cols = num_cols
        self.num_rows = num_rows

    def read_bin_file(self,filename,file_type='np'):
        '''
        Change imaging_types to file_type, unpack is needed when data is from ESPROS; dumped such that it can not be read with np.fromfile
        For data from pythonic code this would be read using np.fromfile
        '''
        with open(filename, mode='rb') as file: ## b is important -> binary
            if file_type == 'unpack':
                    fileContent = file.read()
                    size_of_file = len(fileContent)
                    array_input = struct.unpack('>'+'H'*(size_of_file//2), fileContent)
                    array_input = np.uint16(array_input) ## for original and amplitude images
            elif file_type=='np':
                    array_input = np.fromfile(file,np.uint16()) ## for filtered images
        return array_input
    def read_dist_ampl_files(self, filename, file_type = 'np'):
        '''
        read distance and ampl file; if saved by python we should use fromfile, if saved by espros we can try unpack
        '''
        array_input = self.read_bin_file(filename,file_type)
        
        array_input = array_input[:self.num_cols*self.num_rows].reshape((self.num_rows,self.num_cols))

        return array_input.transpose()
    
        
 
    def read_hdr_files(self, filename, file_type = 'np', imaging_type='dist'):
        '''
        read HDR files, key difference is number of frames, for distance there will be 4 frames
        below; imaging_type needed for that
        '''
        array_input = self.read_bin_file(filename,file_type)

        if imaging_type == 'ampl':
            array_input = array_input[:self.num_cols*self.num_rows].reshape((1,self.num_rows,self.num_cols))
        else:
            array_input = array_input[:self.num_cols*self.num_rows*4].reshape((4,self.num_rows,self.num_cols))

        return np.transpose(array_input,(2,1,0))

    def read_dcs_files(self, filename, file_type = 'np', imaging_type ='dist'):
        '''
        read dcs file using a hack to read 4th component from _ampl files, so read 3 dcs from one file and 1 dcs from ampl file; hence imaging_type needed
        '''
        array_input = self.read_bin_file(filename,file_type)

        # here is pure hack, 4th component being saved in ampl file!! this because of write_file last component is being written to ampl
        if imaging_type == 'dcs4':
            array_input = array_input[:self.num_cols*self.num_rows].reshape((1,self.num_rows,self.num_cols))
        else:
            array_input = array_input[:self.num_cols*self.num_rows*3].reshape((3,self.num_rows,self.num_cols))

        return np.transpose(array_input,(2,1,0))
    

         

    def bin_to_frame(self, input_data_folder_name, file_index,image_no,imaging_type='Dist_Ampl' ):
        '''
        the function that reads and then stacks and adds info so as to make it compliant to streaming format
        both _ampl.bin and .bin are read and according to imaging_type parsed appropriately
        if no bin file return None
        '''
        
        filename = input_data_folder_name+'/'+file_index+'_ampl.bin'
        if not os.path.exists(filename):
            return None
        if imaging_type == 'Dist_Ampl':
            ampl_image_array = self.read_dist_ampl_files(filename)
        elif imaging_type == 'HDR':
            ampl_image_array = self.read_hdr_files(filename, imaging_type = 'ampl')
        elif imaging_type == 'DCS':
            _dcs_image_array = self.read_dcs_files(filename,imaging_type = 'dcs4')

        filename = input_data_folder_name+'/'+file_index+'.bin'
        if not os.path.exists(filename):
            return None
        if imaging_type == 'Dist_Ampl':    
            depth_image_array = self.read_dist_ampl_files(filename)
        elif imaging_type == 'HDR':
            depth_image_array = self.read_hdr_files(filename, imaging_type = 'dist')
        elif imaging_type == 'DCS':
            dcs_image_array = self.read_dcs_files(filename,imaging_type  = 'dist')
            
        # this is being added to make it compliant to streaming format, where dist and ampl are stacked, also notice that return is a list of frames, type, number etc
        if imaging_type == 'Dist_Ampl':
            frame = np.stack((depth_image_array,ampl_image_array),axis=-1)
        elif imaging_type == 'HDR':
            frame = np.dstack((depth_image_array,ampl_image_array))
        elif imaging_type == 'DCS':
            frame = np.dstack((dcs_image_array,_dcs_image_array))
            
        return [frame,"disk",image_no,imaging_type]

    def mat_to_frame(self, input_data_folder_name, file_index):
        '''
        read data from mat file
        '''
        
        _filename = input_data_folder_name+'/'+file_index+'.mat'
        filename = _filename.replace('imageDistance','data')
        try:
            D = scipy.io.loadmat(filename)['data']
            # rgbdata is added as the last element at controller level along with the timestamps
            if len(D[0]) > 4:
                rgbdata = D[0][4]
            else:
                rgbdata = None
                # this is the 2nd element but then its a list byitself coming from epc_image, which has chip and frame number, and imaging_type
            data = D[0][2][0]
            return data, rgbdata
        except FileNotFoundError:
            return None, None
        
