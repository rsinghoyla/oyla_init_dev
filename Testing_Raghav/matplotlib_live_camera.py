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

sys.path.append("../DME_Python_3_Current/")

from epc_lib import *
sys.path.append("../DME_Python_3_Current/oyla")
from utils import plotting_dist_ampl
from models_espros_660 import Camera
def main():
    
    camera = Camera(0)
    
    camera.initialize()
    #live_display = input("Live Display: ")
    live_display = "On"
    save_data = "Off"
    #measurement loop
    iterator = 0
    then = time.time() #Time before the operations start
    numberOfFrames = 50
    imageData3D = np.empty([160*2, 120*2, 2, numberOfFrames], dtype='float')
    while iterator < numberOfFrames:
        print('send server command at ',time.time())
        imageData3D[:,:,:,iterator] = camera.get_frame()
        #print(iterator)
        if live_display == 'On':
            dist_sin=imageData3D[:,:,0,iterator]
            dist_sin = dist_sin.astype('float')
            dist_sin[np.where(dist_sin>30000)] = 0
            dist_sin=(dist_sin/30000.0)
            ampl=imageData3D[:,:,1,iterator]
            ampl= ampl.astype('float')
            #time.sleep(0.01);
            print('update view',np.mean(imageData3D[:,:,:,iterator]),time.time())
            plotting_dist_ampl(dist_sin,None,parameters={'ambiguityDistance':1.0,'cmap_distance':'jet_r'})
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
        # save_path = save_path+'/imageData3D.h5'
        # f = h5py.File(save_path)
        # f['imageData3D'] = imageData3D
        # f.close()
        for i in range(numberOfFrames):
            print(imageData3D[:,:,0,i].shape,imageData3D.dtype)
            with open(save_path+'/imageDistance'+str(i+1)+'.bin','wb') as fp:
                _tmp = bytearray(imageData3D[:,:,0,i].transpose())
                fp.write(_tmp)
            with open(save_path+'/imageDistance'+str(i+1)+'_ampl.bin','wb') as fp:
                _tmp = bytearray(imageData3D[:,:,1,i].transpose())
                fp.write(_tmp)
                    

    print("INFO: Script successfully terminated.")

if __name__ == '__main__':
    main()
