##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Creation:          20.04.2019                                                                  #
# Version:           1.0                                                                         #
# Revision history:  1.1 Initial version                                                         #
# Description
#         all the espros server commands needed for setup before acquistion
##################################################################################################

import sys
sys.path.append("../DME_Python_3_Current/")

from epc_lib import *
class Camera:
    def __init__ (self, chip):

        #####################################################################################
        # initializes the espros server and creates a data structure for acquisition
        # input: chip_ip
        # output: server, image for each chip
        # let servers, image be  dictionary where key is the chip_id
        #####################################################################################
        
        self.server  =  epc_server(chip)  # IP of the evalkit
        self.image = epc_image(self.server)

    def server_commands(self,parameters,epoch_number):
        #####################################################################################
        # executes a sequence of server commands based on parameters and epoch_number
        # input: server to execute on, parameters and epoch_number
        # output: 0 
        # TODO -- maybe we need to catch exceptions here
        #####################################################################################
    
        for p in parameters:
            if p =='setROI' or p == 'SelectShutterMode' or p == 'range_max' or p == 'range_min' or p == 'vertical_flip' or p == 'horizontal_flip' :
                # hack to  ignore these parameters because they return error from server
                continue
            if parameters[p][epoch_number] != '': #if value cell is empty, its not executed
                string = p+" {}".format(parameters[p][epoch_number])
                #print(string)
                self.server.sendCommand(string)

    def logical_server_commands(self,parameters,epoch_number):
    
        #####################################################################################
        # Based on parameters and coding logic a sequence of server commands is executed
        # input: server, parameters, epoch_number, image (for setting dimensions) TODO
        # output: 0 
        # TODO -- maybe we need to catch exceptions here
        #####################################################################################
    

        # irrespetive of imaging_type the imaging_mode is being set;
        # TODO check if this should not be done for Gray imaging type
        # also both MGX and HDR can not be turned on at the same time (from UI)
        if parameters['imaging_mode'][epoch_number] == 'MGX':
            self.server.sendCommand("enableDualMGX 1")
            self.server.sendCommand("enableHDR 0")
        elif parameters['imaging_mode'][epoch_number] == 'Std':
            self.server.sendCommand("enableDualMGX 0")
            self.server.sendCommand("enableHDR 0")
        elif parameters['imaging_mode'][epoch_number] == 'HDR':
            self.server.sendCommand("enableHDR 1")
            #server.sendCommand("enableDualMGX 0")


        # will be executed only if value (1/0)  filled for the epoch_number
        if parameters['enableCompensations'][epoch_number] == '1':
            self.server.sendCommand("correctGrayscaleGain 1")    
            self.server.sendCommand("correctGrayscaleOffset 1")
            self.server.sendCommand("correctTemperature 1")      # 1 = enable temperature correction
            self.server.sendCommand("correctAmbientLight 1")     # 1 = enable ambient light correction
            self.server.sendCommand("correctDRNU 2")             # 2 = enable DRNU correction
        
        elif parameters['enableCompensations'][epoch_number] == '0':
            self.server.sendCommand("correctTemperature 0")      # 1 = enable temperature correction
            self.server.sendCommand("correctAmbientLight 0")     # 1 = enable ambient light correction
            self.server.sendCommand("correctDRNU 0")             # 2 = enable DRNU correction
            #server.sendCommand("correctAmbientLight 0")
            # 0 = disable ambient light correction (there is no correciton available)
            # NOTE: THE COMMAND "getDCSSorted" doesn't have ambient light compensation yet!!!
            self.server.sendCommand("correctGrayscaleGain 0")    # 0 = disable PRNU correction
            self.server.sendCommand("correctGrayscaleOffset 0")  # 0 = disable DSNU correction

        # will be executed only if value (1/0)  filled for the epoch_number
        if parameters['enablePreheat'][epoch_number] == '1':
            # enabling/disabling illumination preheat
            #enable preheat
            self.server.sendCommand("w 90 c4")
            self.server.sendCommand("w ab 00")
        elif parameters['enablePreheat'][epoch_number] == '0': 
	    #disable preheat
            self.server.sendCommand("w 90 cc")
            self.server.sendCommand("w ab 01")

        if parameters['enableHorizontalBinning'][epoch_number] == '1':
            self.server.sendCommand("enableHorizontalBinning 1")
        elif parameters['enableHorizontalBinning'][epoch_number] == '0':
            self.server.sendCommand("enableHorizontalBinning 0")

        if parameters['enableVerticalBinning'][epoch_number] == '1':
            self.server.sendCommand("enableVerticalBinning 1")
        elif parameters['enableVerticalBinning'][epoch_number] == '0':
            self.server.sendCommand("enableVerticalBinning 0")
        
        # TODO -- this was copied from the scripts. Its unclear if fullROI = 0 what commands are send
        #         to server. Also this should be related to setROI but that is throwing an error.
        # if parameters['adaptive_cmd']['setROI'][epoch_number] == '4 323 6 245':
        fullROI = 1
        #else:
        print('only full ROI supported at this point')
        #    return None
    
        if fullROI:
            self.server.sendCommand("w 11 fa")
            icType = self.server.sendCommand("r 12")
            if(icType[0]==2):
                numberOfRows        = 240
                numberOfColumns     = 320
                if parameters['enableHorizontalBinning'][epoch_number] == '1':
                    numberOfRows = 120
                if parameters['enableVerticalBinning'][epoch_number] == '1':
                    numberOfColumns = 160    
            elif (icType[0]==4):
                # Never tested this
                numberOfRows        = 60
                numberOfColumns     = 160
            
            #numberOfRows        = 240
            #numberOfColumns     = 320
        # setting the dimension of the image;
        #     final dimension of numberofdataframes will be set in acquiring.py depending on imaging_type 
        self.image.setNumberOfRecordedColumns(numberOfColumns)
        self.image.setNumberOfRecordedRows(numberOfRows)
        
    def get_frame(self,imaging_type):
        
        #####################################################################################
        # sets up the image in particular the numberOfRecordedImageDataFrames which depend on
        #      image_type, also the function that is returned depends on image_type
        #      also loadConfig is sent as 0 or 1 to server depending upon image type (gray of TOF)
        # input: server, image, imaging_type
        #        imaging_type here are basic imaging type: Gray, DCS, Dist_Ampl, Dist, Ampl
        #                     combinations of this are specified in main function using +
        # output: function that acquires the imaging_tupe signal from image
        #####################################################################################
        self.last_frame = None
        if imaging_type == 'Gray':
            # this setting is followin the setting in *2D_raw4DCS* scripts

            self.server.sendCommand("loadConfig 0")     # loadConfig 0 = Grayscale mode (2D imaging)
            numberOf2Dimage_DCS = 1   
            self.image.setNumberOfRecordedImageDataFrames(numberOf2Dimage_DCS)
            self.image.updateNbrRecordedBytes()
            self.last_frame =  self.image.getDCSs() #Because

        elif imaging_type == 'Dist_Ampl':
            # this setting is followin the setting in *3D_distanceAmplitude* scripts

            self.server.sendCommand("loadConfig 1")                  # loadConfig 1 = TOF mode (3D imaging)
            numberOf3DimageDataframe = 2
            self.image.setNumberOfRecordedImageDataFrames(numberOf3DimageDataframe);
            self.image.updateNbrRecordedBytes()
            self.last_frame =  self.image.getDistAmpl()   # acquire Distance

        elif imaging_type == 'HDR':
            # this setting is followin the setting in *3D_distanceAmplitude* scripts

            self.server.sendCommand("loadConfig 1")                  # loadConfig 1 = TOF mode (3D imaging)
            numberOf3DimageDataframe = 4
            self.image.setNumberOfRecordedImageDataFrames(numberOf3DimageDataframe);
            self.image.updateNbrRecordedBytes()
            self.last_frame =  self.image.getDistAmplHDR()   # acquire Distance 

        elif imaging_type == 'Dist':

            self.server.sendCommand("loadConfig 1")                  # loadConfig 1 = TOF mode (3D imaging)
            numberOf3DimageDataframe = 1
            self.image.setNumberOfRecordedImageDataFrames(numberOf3DimageDataframe);
            self.image.updateNbrRecordedBytes()
            self.last_frame =  self.image.getDist()   # acquire Amplitude

        elif imaging_type == 'Ampl':

            self.server.sendCommand("loadConfig 1")                  # loadConfig 1 = TOF mode (3D imaging)
            numberOf3DimageDataframe = 1
            self.image.setNumberOfRecordedImageDataFrames(numberOf3DimageDataframe);
            self.image.updateNbrRecordedBytes()
            self.last_frame =  self.image.getAmpl()   # acquire Distance + Amplitude

        elif imaging_type == 'DCS':
            # this setting is followin the setting in *2D_raw4DCS* scripts

            self.server.sendCommand("loadConfig 1")              # loadConfig 1 = TOF mode (3D imaging)
            numberOf3DimageDataframe = 4
            self.image.setNumberOfRecordedImageDataFrames(numberOf3DimageDataframe);
            self.image.updateNbrRecordedBytes()
            self.last_frame =  self.image.getDCSs()   # grap DCS's

        elif imaging_type == 'DCS_2':
            # this setting is followin the setting in *2D_raw4DCS* scripts

            self.server.sendCommand("loadConfig 1")              # loadConfig 1 = TOF mode (3D imaging)
            numberOf3DimageDataframe = 2
            self.image.setNumberOfRecordedImageDataFrames(numberOf3DimageDataframe);
            self.image.updateNbrRecordedBytes()
            self.last_frame =  self.image.getDCSs()   # grap DCS's
        return self.last_frame
