import numpy as np
import sys
sys.path.append('../DME_Python_3_Current')
from epc_lib import *
import time

class Camera:
    def __init__(self, cam_num):
        self.cam_num = cam_num
        self.cap = None
        self.server  = epc_server("192.168.7.2")     # IP of ssh over USB
        self.image_class = epc_image(self.server)
        self.last_frame = np.zeros((1,1))


    def initialize(self):
        #####################################################################################
        # Parameters                                                                        #
        #####################################################################################
        server1 = self.server
        image1 = self.image_class
        
        fullROI                 =   1   #1 for full ROI image
        enableCompensations     =   1  #1 for compensated DATA
        enableCompensations     =   1 #1 for compensated DATA
        enablePreheat           =   0   # 0 - disable preheat
        if fullROI==1:
            server1.sendCommand("w 11 fa")
            icType=server1.sendCommand("r 12")
        
            if(icType[0]==2):
                numberOfRows        = 240
                numberOfColumns     = 320
                numberOf3DimageDataframe = 2     # 4 DCS acquisition
            elif (icType[0]==4):
                numberOfRows        = 60
                numberOfColumns     = 160
                numberOf3DimageDataframe = 2     # 4 DCS acquisition
                # else:
                #     numberOfColumns     = int(input("Enter number of cols:"))
                #     numberOfRows        = int(input("Enter number of rows:"))
                #     numberOf3DimageDataframe = 2     # 1 Distance + 1 Amplitued = 2 images

        elif fullROI == 2:
            numberOfRows = 120
            numberOfColumns = 160
            numberOf3DimageDataframe = 2     # 4 DCS acquisition
    

        #####################################################################################
        # initialize var                                                                    #
        #####################################################################################

        #imageData3D = np.empty([numberOfColumns, numberOfRows, numberOf3DimageDataframe, self.numberOfFrames], dtype='uint16')
        #temperatureData     = np.empty([self.numberOfFrames], dtype='int16')

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
        server1.sendCommand("selectShutterMode 1")           # shutterMode 0 = I2C shutter (default value), shutterMode 1 = BBB GPIO HW shutter, shutter 2 = external shutter must be applied
        server1.sendCommand("setModulationFrequency 1")
        server1.sendCommand("enableHDR 0")
        if fullROI == 1:
            server1.sendCommand("enableVerticalBinning 0")
            server1.sendCommand("enableHorizontalBinning 0")
        elif fullROI ==2:
            server1.sendCommand("enableVerticalBinning 1")
            server1.sendCommand("enableHorizontalBinning 1")
        server1.sendCommand("enablePiDelay 1")
        
        #self.cap =  self.image_class.getDistAmpl()   # acquire Distance

    def get_frame(self):
        # self.server.sendCommand("loadConfig 0")     # loadConfig 0 = Grayscale mode (2D imaging)
        # numberOf2Dimage_DCS = 1   
        # self.image_class.setNumberOfRecordedImageDataFrames(numberOf2Dimage_DCS)
        # self.image_class.updateNbrRecordedBytes()
        # self.last_frame =  self.image_class.getDCSs() #Because

        #self.last_frame = self.image_class.getDistAmpl()
        
        numberOf3DimageDataframe = 4
        self.image_class.setNumberOfRecordedImageDataFrames(numberOf3DimageDataframe);
        self.image_class.updateNbrRecordedBytes()
        self.last_frame =  self.image_class.getDistAmplHDR()   # acquire Distance 
        print('got image',np.mean(self.last_frame),time.time())
        #b = np.repeat(self.last_frame[:, :, 0][:,:,np.newaxis], 3, axis=2)
        self.last_frame = self.last_frame.astype('float')
        #self.last_frame[:,:,0][np.where(self.last_frame[:,:,0]>60000)] = 0
        #self.last_frame[:,:,0] = self.last_frame[:,:,0]/30000.0
        #self.last_frame[:,:,1][np.where(self.last_frame[:,:,1]>60000)] = 0
        #print(self.last_frame.shape)
        #print(np.mean(self.last_frame[:,:,0]))
        return self.last_frame

    # def acquire_movie(self, num_frames):
    #     import time
    #     then = time.time()
    #     movie = []
    #     for _ in range(num_frames):
    #         print('acquiring for movie',time.time())
    #         movie.append(self.get_frame())
    #         #print(np.mean(movie[-1]))
    #     now = time.time()
    #     print("Frame rate: ", (num_frames/(now-then)))
    #     return movie


    #def close_camera(self):
    #    self.cap.release()

    def __str__(self):
        return 'ESPROS Camera {}'.format(self.cam_num)


if __name__ == '__main__':
    cam = Camera(0)
    cam.initialize()
    print(cam)
    frame = cam.get_frame()
    print(frame)
    #cam.set_brightness(1)
    #print(cam.get_brightness())
    #cam.set_brightness(0.5)
    #print(cam.get_brightness())
    #cam.close_camera()
