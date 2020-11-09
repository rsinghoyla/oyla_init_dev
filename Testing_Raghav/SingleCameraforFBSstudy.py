##################################################################################################
#                                                                                                #
# Author:            Sid                                                                       #
# Creation:          22.04.2019                                                                  #
# Version:           1.0                                                                         #
# Revision history:  1.0 Initial version                                                         #
##################################################################################################


import sys
import h5py
import numpy as np
import time
import datetime
import random
import matplotlib.pyplot as plt

#sys.path.append("../DME_Python_3_Current/")

from epc_lib import *
#sys.path.append("../DME_Python_3_Current/oyla")
from utils_black import plotting_dist_ampl

def main():

    #####################################################################################
    # Main Objects                                                                      #
    #####################################################################################
    #server  = epc_server("192.168.7.2")     # IP of ssh over USB

    server1  =  epc_server("192.168.7.2")  # IP of the evalkit
    image1   = epc_image(server1)

    
    #####################################################################################
    # Parameters                                                                        #
    #####################################################################################

    fullROI                 =   1   #1 for full ROI image
    enableCompensations     =   1  #1 for compensated DATA
    enablePreheat           =   0   # 0 - disable preheat
    live_display = "On"#input("Live Display On/Off: ")
    save_data = "Off"#input("Save Data On/Off:")
    numberOfFrames           = 10#int(input("Enter number measured frames: "))
    #server1.sendCommand('setROI 4 323 6 245')
    if fullROI==1:
        server1.sendCommand("w 11 fa")
        icType=server1.sendCommand("r 12")

        if(icType[0]==2):
            numberOfRows        = 120*2
            numberOfColumns     = 160*2
            numberOf3DimageDataframe = 2     # 4 DCS acquisition
        elif (icType[0]==4):
            numberOfRows        = 60
            numberOfColumns     = 160
            numberOf3DimageDataframe = 2     # 4 DCS acquisition
    elif fullROI==2:
        numberOfRows        = 120
        numberOfColumns     = 160
        numberOf3DimageDataframe = 2     # 1 Distance + 1 Amplitued = 2 images  
        server1.sendCommand('setROI 4 163 66 125')
        
    else:
        numberOfColumns     = int(input("Enter number of cols:"))
        numberOfRows        = int(input("Enter number of rows:"))
        numberOf3DimageDataframe = 2     # 1 Distance + 1 Amplitued = 2 images

    
    

    #####################################################################################
    # initialize var                                                                    #
    #####################################################################################

    imageData3D = np.empty([numberOfColumns, numberOfRows, numberOf3DimageDataframe, numberOfFrames], dtype='uint16')
    temperatureData     = np.empty([numberOfFrames], dtype='int16')

    image1.setNumberOfRecordedColumns(numberOfColumns)
    image1.setNumberOfRecordedRows(numberOfRows)
    image1.setNumberOfRecordedImageDataFrames(numberOf3DimageDataframe);
    image1.updateNbrRecordedBytes()

    #####################################################################################
    # measurement                                                                       #
    #####################################################################################

    
    server1.sendCommand("enableSaturation 1")        # 1 = enable saturation     flag value = 65400
    server1.sendCommand("enableAdcOverflow 1")       # 1 = enable ADC overflow   flag value = 65500

    if  enableCompensations:
        server1.sendCommand("correctTemperature 1")      # 1 = enable temperature correction
        server1.sendCommand("correctAmbientLight 1")     # 1 = enable ambient light correction
        server1.sendCommand("correctDRNU 2")             # 2 = enable DRNU correction

    else:
        server1.sendCommand("correctTemperature 0")      # 1 = enable temperature correction
        server1.sendCommand("correctAmbientLight 0")     # 1 = enable ambient light correction
        server1.sendCommand("correctDRNU 0")             # 2 = enable DRNU correction

    if not enablePreheat:                                   # enabling/disabling illumination preheat
        #disable preheat
        server1.sendCommand("w 90 c4")
        server1.sendCommand("w ab 00")

    else:
        server1.sendCommand("w 90 cc")
        server1.sendCommand("w ab 01")

    server1.sendCommand("setIntegrationTime3D 2000")     # t_int in us
    server1.sendCommand("loadConfig 1")                  # loadConfig 1 = TOF mode (3D imaging)
    #server1.sendCommand("selectShutterMode 1")           # shutterMode 0 = I2C shutter (default value), shutterMode 1 = BBB GPIO HW shutter, shutter 2 = external shutter must be applied
    server1.sendCommand("setModulationFrequency 1")
    server1.sendCommand("enableHDR 0")
     
    server1.sendCommand("enableVerticalBinning 0")
    server1.sendCommand("enableHorizontalBinning 0")
    server1.sendCommand("enablePiDelay 0")
   
    #measurement loop
    iterator = 0
    then = time.time() #Time before the operations start
 
    while iterator < numberOfFrames:
        imageData3D[:,:,:,iterator] = image1.getDistAmpl()   # acquire Distance + Amplitude
        print(iterator,imageData3D.shape)
        if live_display == 'On':
            dist_sin=imageData3D[:,:,0,iterator]
            dist_sin = dist_sin.astype('float')
            dist_sin[np.where(dist_sin>30000)] = 0
            dist_sin=(dist_sin/30000.0)
            ampl=imageData3D[:,:,1,iterator]
            ampl= ampl.astype('float')
            print(dist_sin.shape)
            dist_sin = dist_sin.transpose()
            ampl = ampl.transpose()
            dist_sin = np.flipud(np.fliplr(dist_sin))
            #time.sleep(0.01);
            plotting_dist_ampl(dist_sin,ampl,parameters={'ambiguityDistance':1.0,'cmap_distance':'jet_r'})
            plt.pause(0.0000001)
            plt.clf()
            
            
        iterator +=1;
    now = time.time() #Time after it finished
    print("Frame rate: ", (numberOfFrames/(now-then)))

    #save files loop
    #f = h5py.File(('../measData/imageData3D.h5'), 'w')
    #f['imageData3D'] = imageData3D
    #f.close()
    if save_data == 'On':
        save_path = './'
        t = datetime.datetime.now()    
        save_path = save_path+'_{:%d_%H_%M_%S}'.format(t)
        if not os.path.isdir(save_path): os.mkdir(save_path)
        save_path = save_path+'/imageData3D.h5'
        f = h5py.File(save_path)
        f['imageData3D'] = imageData3D
        f.close()
        # for i in range(numberOfFrames):
        #     print(imageData3D[:,:,0,i].shape,imageData3D.dtype)
        #     with open(save_path+'/imageDistance'+str(i+1)+'.bin','wb') as fp:
        #         imageData3D[:,:,0,i].tofile(fp)
        #     with open(save_path+'/imageDistance'+str(i+1)+'_ampl.bin','wb') as fp:
        #         imageData3D[:,:,1,i].tofile(fp)
                    

    print("INFO: Script successfully terminated.")

if __name__ == '__main__':
    main()
