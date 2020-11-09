import struct  ## library for unpacking of bin file

import numpy as np ## numpy array

import matplotlib.pyplot as plt ## for saving image 

from models_espros_660 import Camera

import sys ## for time and system call

import time



import vispy

from vispy import app

from vispy import scene , visuals  

from vispy.color import get_colormap, Colormap

SPEED_OF_LIGHT = 300000000.0
MAX_PHASE = 30000
        
class Canvas(app.Canvas):
    def __init__(self):
        
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor = 'white')
        
        self.canvas.unfreeze()
        
        self.canvas.view = self.canvas.central_widget.add_view()
        
        #self.canvas.view.camera = 'arcball'
        
        #self.canvas.Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
        
        #self.canvas.scatter = scene.visuals.Markers(parent = self.canvas.view.scene )
        
        
        
        #self.canvas.view.add(self.canvas.scatter)

        self.canvas.grid = self.canvas.central_widget.add_grid(margin=10)

        colormap = Colormap(['r', 'g', 'b'])

        colormap[np.linspace(0., 1., 100)]

        self.canvas.cbar_widget = scene.ColorBarWidget(label="Depth color scale", clim=(range_min, range_max),
                                           cmap=colormap, orientation="left", border_width=1)

        self.canvas.grid.add_widget(self.canvas.cbar_widget,col = 10)

        self.canvas.cbar_widget.border_color = "#212121"
        self.canvas.image = scene.visuals.Image(parent = self.canvas.view.scene,cmap=colormap,clim=(range_min, range_max))
        
        self.canvas.freeze()
        
        self._timer = app.Timer(0.0000001, connect=self.on_timer, start=True,iterations=total_num_of_images)
        
        #self.draw_bounding_box()
            
    
        
    def on_timer(self, event):
        start_of_update = time.time()
        global img_no
        global camera

        global transform_type

        global remove

        global mod_freq

        frame = camera.get_frame()

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
        
        #frame = np.flipud(np.transpose(camera.get_frame()[:,:,0]))
        
        depth_image_array = np.rint((int((SPEED_OF_LIGHT/2)/mod_freq*1000)*(depth_image_array/MAX_PHASE))/10)
        
        if img_no == 1:
            self.then = start_of_update
##        print(depth_image_array.shape)
        ## set pos and color array data to scatter markers.

        self.canvas.image.set_data(np.rot90(depth_image_array))
        self.canvas.view.camera = scene.PanZoomCamera(aspect=1)
        self.canvas.view.camera.set_range()
        self.canvas.view.add(self.canvas.image)

        #self.canvas.view.camera = 'turntable'  ## or try 'arcball'

        img_no = img_no + 1
        
        if(img_no > total_num_of_images):
            now = time.time()
            print("Frame rate: ", (total_num_of_images/(now-self.then)))

            #stop()
            #img_no = 1
        

    
        end_of_update = time.time()
        
##        print("Total time taken for one update in milli sec", 1000*(end_of_update - start_of_update))
##        print(img_no)
        self.canvas.update()
            




img_no = 1
mod_freq = 12000000
range_max = 600

range_min = 40
camera = Camera(0)
camera.initialize()
total_num_of_images = 50


if __name__ == '__main__':

    canvas = Canvas()
    
    if sys.flags.interactive == 0:

        app.run()
