#https://stackoverflow.com/questions/53032042/live-plotting-of-many-subplots-using-pyqtgraph
import cv2
from PyQt5.QtCore import (QThread, Qt, pyqtSignal)
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel)
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtGui
import sys
from models_espros_660 import Camera
import numpy as np
#import qimage2ndarray
COLORTABLE=[]
for i in range(256): COLORTABLE.append(QtGui.qRgb(255-i,i,i))
import time
SPEED_OF_LIGHT = 300000000.0
MAX_PHASE = 30000

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    camera = Camera(0)
    camera.initialize()
    def run(self):
        #cap = cv2.VideoCapture(0)
        count = 0
        then = time.time()
        global mod_freq
        while count< 50:
            #ret, frame = cap.read()
            count += 1
            print('a')
            frame = self.camera.get_frame()
            #print('g',np.mean(frame))
            saturation_flag = False
            adc_flag = False
        
            if saturation_flag or adc_flag:
                ampl_image_array = frame[:,:,1]
    
    
            ## calling  bin_to_depth_image function to convert bin image to depth image by unpacking binary value   
            depth_image_array = frame[:,:,0]

    
            if saturation_flag:
                depth_image_array[np.where(ampl_image_array==65400)] = 0
            if adc_flag:
                depth_image_array[np.where(ampl_image_array==65500)] = 0
    
        
            depth_image_array = np.rint((int((SPEED_OF_LIGHT/2)/mod_freq*1000)*(depth_image_array/MAX_PHASE))/10)
            rgbImage = depth_image_array
            if np.any(rgbImage):
                # https://stackoverflow.com/a/55468544/6622587
                #rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #    im_np = rgbImage.copy()#np.ones((1800,2880,3),dtype=np.uint8)                                                                                                                                                                                  
                #    im_np = np.transpose(im_np, (1,0)).copy()
                rgbImage = np.uint8(rgbImage/700*255)
                rgbImage = np.transpose(rgbImage)
                im_np  = np.require(rgbImage,np.uint8,'C')
                #print(np.max(im_np),np.min(im_np),np.max(rgbImage),np.min(rgbImage))
                convertToQtFormat = QImage(im_np.data, im_np.shape[1], im_np.shape[0],QImage.Format_Indexed8)
                #convertToQtFormat = qimage2ndarray.array2qimage(rgbImage)
                
                convertToQtFormat.setColorTable(COLORTABLE)
                
                p = convertToQtFormat#.scaled(640, 480, Qt.KeepAspectRatio)
                print('e')
                self.changePixmap.emit(p)
                
        now = time.time()
        print("Frame rate: ", (50/(now-then)))


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt4 Video'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        print('u')
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1800, 1200)
        # create a label
        self.label = QLabel(self)
        self.label.move(280, 120)
        self.label.resize(640, 480)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()

mod_freq = 12000000
range_max = 600
range_min = 40

if __name__ == '__main__':
    app = QApplication([])
    ex = App()
    ex.show()
    sys.exit(app.exec_())
