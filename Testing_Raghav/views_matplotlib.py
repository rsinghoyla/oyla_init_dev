import numpy as np

from PyQt5.QtCore import Qt, QThread, QTimer,QRectF
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QSlider
#from pyqtgraph import ImageView
import pyqtgraph as pg
from matplotlib import cm
from mpl_cmaps_in_ImageItem import cmapToColormap
from matplotlib import pyplot as plt
import time
class StartWindow(QMainWindow):
    def __init__(self, camera = None,number_images=4,colormaps=[]):
        super().__init__()
        self.camera = camera

        #self.central_widget = QWidget()
        #self.button_frame = QPushButton('Acquire Frame', self.central_widget)
        #self.button_movie = QPushButton('Start Movie', self.central_widget)
        #self.image_view = ImageView()
        #self.image_view.ui.histogram.hide()
        self.number_images = number_images
        #self.canvas =  pg.GraphicsLayoutWidget()
        #self.colormaps = colormaps
        # colormap = cm.get_cmap("jet_r")  # cm.get_cmap("CMRmap")
        # colormap._init()
        # lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # colormap = cm.get_cmap("gray")  # cm.get_cmap("CMRmap")
        # colormap._init()
        # lut_g = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # if len(self.colormaps) != self.number_images:
        #     # Convert a matplotlib colormap into a list of (tickmark, (r,g,b)) tuples
            
            
        #     self.colormaps = []

        #     for i in range(number_images):
        #         if i == 0:
        #             pos, rgba_colors = zip(*cmapToColormap(cm.jet_r))
        #             # Set the colormap
        #             #print(pos,rgba_colors)
        #             pgColormap =  pg.ColorMap(pos, rgba_colors)
        #         else:
        #             pos, rgba_colors = zip(*cmapToColormap(cm.gray))
        #             # Set the colormap
        #             pgColormap =  pg.ColorMap(pos, rgba_colors)
                    
        #         self.colormaps.append(pgColormap)
                
        
        # self.image_views = []
        # for i in range(self.number_images):
        #     view = self.canvas.addPlot()
        #     view.hideAxis('bottom')
        #     view.hideAxis('left')
        #     image_view = pg.ImageItem(None,border="w")
        #     view.addItem(image_view)
        #     #view.getViewBox().setAspectLocked(True)
            
        #     # Apply the colormap
            
        #     #image_view.setLookupTable(self.colormaps[i].getLookupTable())
        #     hist = pg.HistogramLUTItem()
        #     #hist.gradient.loadPreset('spectrum')
            
        #     hist.gradient.setColorMap(self.colormaps[i])
        #     dcc = hist.gradient.saveState()
        #     #print(dcc)
        #     # Link the histogram to the image
        #     hist.setImageItem(image_view)
        #     # for tick in hist.gradient.ticks:
        #     #     tick.hide()
        #     self.canvas.addItem(hist)

        #     #view.addItem(hist)
        #     #view.getViewBox().setAspectLocked(True)

            
            
        #     #view.setRange(QRectF(0, 0, 720, 1280))
        #     #view.setAspectLocked(True)
        #     self.image_views.append(image_view)
        #     if i%2:
        #         self.canvas.nextRow()
        
        # self.layout = QVBoxLayout(self.central_widget)
        # self.layout.addWidget(self.button_frame)
        # self.layout.addWidget(self.button_movie)
        # #self.layout.addWidget(self.image_view)
        # self.layout.addWidget(self.canvas)
        # self.setCentralWidget(self.central_widget)
        # #self.central_widget.resize(1280,720)
        
        # self.button_frame.clicked.connect(self.update_image)
        # self.button_movie.clicked.connect(self.start_movie)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_movie)
        self.start_movie()
        plt.show()


    def update_image(self):
        frame = self.camera.get_frame()
        #frame = np.transpose(frame,axes=(1,0,2))
        #for i,im in enumerate(self.image_views):
        #    im.setImage(np.fliplr(frame[:,:,i]))
        plt.clf()
        plt.imshow(np.fliplr(frame[:,:,0]))
        #plt.pause(0.0001)
        
    def update_movie(self):
        frame = self.camera.last_frame
        print('update view',np.mean(frame),time.time())
        if np.sum(frame.shape) > 2 and frame.ndim ==3 :
            #frame = np.transpose(frame,axes=(1,0,2))
            #for i,im in enumerate(self.image_views):
            #    im.setImage(np.fliplr(frame[:,:,i]))
            plt.imshow(np.fliplr(frame[:,:,0]),cmap='gray')
            plt.pause(0.001)
    def start_movie(self):
        self.movie_thread = MovieThread(self.camera)
        self.movie_thread.start()
        self.update_timer.start(30)


class MovieThread(QThread):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera

    def run(self):
        then = time.time()
        num_frames = 100
        for _ in range(num_frames):
            print('send server command at ',time.time())
            data = self.camera.get_frame()#sensor_data(4, 10, 10)
        now = time.time()
        print("Frame rate: ", (num_frames/(now-then)))
        import sys
        sys.exit()

if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.resize(800,800)
    window.show()
    app.exit(app.exec_())
