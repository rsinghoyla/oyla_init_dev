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
from oyla.mvc.utils import transformation3, phase_to_distance, getComparedHDRAmplitudeImage_vectorized,camera_calibrations
from oyla.mvc.utils import threshold_filter,threshold_coordinates, FOV, CAMERA_VERSION
from vispy.scene.visuals import Text
from oyla.mvc.filters import filter_temporal_median

class Canvas(scene.SceneCanvas):
    """
    Canvas class set the parameter value that comes from Widget window and render pointcloud on the vispy canvas.
    Initialization -- canvas of size (800,800), camera as turntable, scatter as visual marker, 
    grid to hold colorbar representing the depth distance.
                draw _bounding_box function draw line using Plot3D
                on_timer function is called repeatedly and pointcloud is updated and displayed on vispy canvas 
                on_mouse_move function update the value of angle and scale value on widget window by checking the camera state

    on_timer calls the params_update function to update the parameter values repeatedly
    """
    def __init__(self, azimuth = None, elevation = None,  x_translate=None, y_translate=None, z_translate=None,
                 scale=None, bounding_box=False, colormap_type = 'rgb', view_type = 'default_view',
                 transform_type ='cartesian', camera_type = 'turntable',fov_angle_x = None, fov_angle_y = None,
                 setup_widget = None,
                 range_max = None, range_min = None, x_max = None, x_min = None, y_max= None, y_min = None,rgb = False,ampl_min = 50,
                 reflectivity_thresh = False,colorize = False, bbox = None,qp_ampl = 1, qp_phase = 1):
        
        #app.Canvas.__init__(self, title='Widget_Controller', keys='interactive', size=(650, 650))
        scene.SceneCanvas.__init__(self, keys='interactive', size=(1000, 1000), show=False, bgcolor = 'black')
        # Some initialization constants that won't change; nearly everything that may change has to be defined with some
        # default value here while Canvas is unfrozen
        self.unfreeze()

        # the parameter widget is passed here so that the mouse events gets reflected back in text field
        self.rgb  = rgb
        self.setup_widget  = setup_widget
        
        self.set_parameters( azimuth = azimuth, elevation = elevation,
                             x_translate=x_translate, y_translate=y_translate,
                             z_translate=z_translate,scale=scale, bounding_box=bounding_box,
                             colormap_type = colormap_type, view_type = view_type,
                             transform_type = transform_type, fov_angle_x = fov_angle_x, fov_angle_y = fov_angle_y,
                             camera_type = camera_type,
                             range_max = range_max, range_min = range_min, x_max = x_max, x_min = x_min,
                             y_max= y_max, y_min = y_min, ampl_min = ampl_min, reflectivity_thresh = reflectivity_thresh, colorize = colorize,
                             qp_ampl = qp_ampl, qp_phase=qp_phase)
        
        self.grid = self.central_widget.add_grid(margin=10)
        
        self.view = self.grid.add_view(row=0,col=0,row_span=3,col_span=14)#scene.widgets.ViewBox(border_color= None, parent=self.scene)#
        self.view_cb = self.grid.add_view(row=0,col=15,row_span=3)
        # this was hand coded to ensure colorbar, pcl view, rgb view are added
        if self.rgb:
            #self.view_dummy = self.grid.add_view(row=0,col=11,col_span=9)
            self.view_rgb = self.grid.add_view(row=4,col=4,col_span=8,row_span=2)
        #self.view = self.central_widget.add_view()
        #import vispy
        self.view.camera = camera_type
        #print(self.view.camera.get_state())
        # the 3D plot scene to be added to view
        self.Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
        self.scatter = scene.visuals.Markers(parent = self.view.scene)
        
        self.view.add(self.scatter)
        self.textVis = Text('', parent=self.scene, color='red')
                    #t1.font_size = 24
        self.textVis.pos =50, 300
        
        # TODO: no colorbar for pcl
        self.cbar = scene.ColorBarWidget(label="", clim=(self.range_min, self.range_max),
                                                cmap="jet_r", orientation="left", border_width=1)
        #self.cbar = scene.visuals.ColorBar(label_str="XYX",size=(640,10),clim=(self.range_min, self.range_max),
        #                                   cmap="jet_r", orientation="left", border_width=1)
        #self.grid.add_widget(self.cbar_widget,row = 0,col = 10,row_span=2)
        self.view_cb.add_widget(self.cbar)
        self.cbar.border_color = "#212121"

        
        #_widget = scene.widgets.ViewBox(border_color=None, parent=self.scene)
        # instead of adding a widget the visual is a child to view_rgb; again how and why this works is not very clear
        if self.rgb :
            #_widget = self.grid.add_view(row=1,col = 11,col_span = 10)
            self.rgb_image = scene.visuals.Image(parent = self.view_rgb.scene)
            self.got_image = False
            #self.view_rgb.camera.set_range(x=(0,640),y=(0,480))
            #self.rgb_image.set_data(np.ones((640,480)))
            #self.view_rgb.camera = scene.PanZoomCamera(aspect=1)

        #For drawing bounding box
        self.pos_listxy = [0,0,0,0,0,0,0,0]
        self.line_listxy = [0,0,0,0,0,0,0,0]
        self.line_listz = [0,0,0,0]
        self.pos_listz = [0,0,0,0]

        #initial params that will get overwritten
        self.number_chips = 1
        self.ambiguity_distance = None
        #self.range_max = 600
        #self.range_min = 40
        self.saturation_flag = True
        self.adc_flag = True
        # all members of canvas class have to be initialized here when canvas is unfrozen
      
        self.interpolation = False
        self.filter_params = {}
        self.mod_freq = None
        self.camera_version = 'oyla_2_camera'
        self.txt_to_be_displayed = None
        self.median_array = None
        self.rgb_array = None
        self.total_num_of_images = 0
        # Append all the visuals
        self.freeze()
        # Set up a timer to update the image and give a real-time rendering
        #self.show()
        
    def update_canvas(self,number_chips = 1, ambiguity_distance = None,
                      range_max = 600 ,range_min = 40, saturation_flag = None ,
                      adc_flag = None, mod_freq = None, filter_params = {},rgb = False,ampl_min = 50, reflectivity_thresh = False, colorize = False,
                      camera_version = 'oyla_2_camera', bbox = None, txt_file = None, total_num_of_images = 0):
        # after initialization update before plotting, this was useful for multiple epochs. 
        self.ambiguity_distance = ambiguity_distance
        self.range_max = range_max
        self.range_min = range_min
        self.saturation_flag = saturation_flag
        self.adc_flag = adc_flag
        self.number_chips = number_chips
        self.mod_freq = mod_freq
        self.filter_params = filter_params
        if bbox is None:
            self.draw_axis()
        else:
            bbox = bbox[0].split(' ')
            print(bbox)
            bbox = map(int,bbox)
            bbox = list(bbox)
            print(bbox)
            self.draw_bounding_box(bbox=bbox)
        if txt_file is not None:
            _tmp = open(txt_file).read().splitlines()
            self.txt_to_be_displayed = {}
            for _t in _tmp:
                __t = _t.split(',')
                self.txt_to_be_displayed[__t[0]] = __t[1]
        else:
            self.txt_to_be_displayed = None
        self.setup_widget.psets[8].setValue(range_max)
        self.setup_widget.psets[9].setValue(range_min)
        self.setup_widget.psets[10].setValue(ampl_min)
        self.rgb = rgb
        self.ampl_min = ampl_min
        self.reflectivity_thresh = reflectivity_thresh
        self.setup_widget.reflectivity_thresh_chk.setChecked(reflectivity_thresh)
        self.colorize = colorize
        self.setup_widget.colorize_chk.setChecked(colorize)
        self.camera_version = camera_version
        self.setup_widget.psets[6].setValue(FOV[CAMERA_VERSION.index(self.camera_version)][1])
        self.setup_widget.psets[7].setValue(FOV[CAMERA_VERSION.index(self.camera_version)][0])
        self.total_num_of_images = total_num_of_images
    def draw_axis(self):
        '''draw bounding box of dim 320, 240, range'''
        # ToDo change 320, 240 into actual dimensions
        box_corner_X = np.array([0,320*self.number_chips,320*self.number_chips,0,0,320*self.number_chips,320*self.number_chips,0,])
        box_corner_Y = np.array([0,0,240,240,0,0,240,240])
        box_corner_Z = np.array([0,0,0,0,self.range_max,self.range_max,self.range_max,self.range_max])

        depth_list = [0]
        color_list = ['red','green','red','green','red','green','red','green']
        for k in depth_list:
            for i in [0,3]:              
                self.pos_listxy[i+k] = np.array([[box_corner_X[i+k],box_corner_Y[i+k],box_corner_Z[i+k]], [box_corner_X[(i+1)%4 + k],box_corner_Y[(i+1)%4 + k],box_corner_Z[(i+1)%4 + k]]])
                self.line_listxy[i+k] = self.Plot3D(self.pos_listxy[i+k], width=1.0, color=color_list[i+k], edge_color='w', symbol='o', face_color=(0.2, 0.2, 1, 0.8), parent=self.view.scene)
        
        self.line_listxy[0].set_data(width=6.0)
        self.line_listxy[3].set_data(width=6.0)

        for i in range(1):
            self.pos_listz[i] =   np.array([[box_corner_X[i],box_corner_Y[i],box_corner_Z[i]], [box_corner_X[4+i],box_corner_Y[4+i],box_corner_Z[4+i]] ])
            self.line_listz[i]  = self.Plot3D(self.pos_listz[i], width=1.0, color='blue',edge_color='w', symbol='o', face_color=(0.2, 0.2, 1, 1),parent=self.view.scene)
        
        self.line_listz[0].set_data(width=6.0)
        
    def draw_bounding_box(self,bbox):
        '''draw bounding box of dim 320, 240, range'''
        # ToDo change 320, 240 into actual dimensions

        seeds = [[0,0,0],[1,1,0],[0,1,1],[1,0,1]]
        for s in seeds:

            for k in range(3):
                _s = s.copy()
                _s[k] = 1-_s[k]
                #print(np.asarray(_s)+np.asarray([0,1,2]))
                start = np.asarray(bbox)[3*np.asarray(s)+np.asarray([0,1,2])]
                end  = np.asarray(bbox)[3*np.asarray(_s)+np.asarray([0,1,2])]
                _ = self.Plot3D([start, end], width = 1.0, color = 'red',edge_color='w', symbol='o', face_color=(0.2, 0.2, 1, 1),parent=self.view.scene)

    def on_mouse_press(self,event):
        print("doing nothing on mouse press ---------------------")
        #print(self.view.camera.get_state())
        #print(event.pos)

    def on_mouse_move(self,event):
        '''on_mouse_move function update the value of angle and scale value on widget window by checking the camera state'''

        self.setup_widget.psets[0].setValue(self.view.camera.azimuth)
        self.setup_widget.psets[1].setValue(self.view.camera.elevation)
        #self.setup_widget.psets[2].setValue(self.view.camera.roll)
        #print("on mouse move --------------------------------------------------------")
        #print(self.view.camera.get_state())
    
    def on_mouse_wheel(self, event):
        
        self.setup_widget.psets[5].setValue(self.view.camera.scale_factor)
        #print('-------------------------Mouse wheel-----------------------')    
        #print(self.view.camera.get_state())


    def set_parameters(self, azimuth = None, elevation = None,  x_translate=None,y_translate=None,z_translate=None,
                       scale=None, bounding_box=False, colormap_type = 'grays', view_type = 'default_view',
                       transform_type = 'cartesian', fov_angle_x = 94, fov_angle_y = 94,camera_type = 'turntable',
                       range_max = None, range_min = None, x_max = None, x_min = None, y_max= None, y_min = None, ampl_min = None,
                       reflectivity_thresh = False,colorize = False, bbox = None,txt_file = None, qp_ampl = 1, qp_phase = 1):
        """
        Initialize parameters for the system that will be used later.
        again the reason for having this is to allow controller to set these parameters. This is different from the parameters set at top which are only used for 
        initialization
        """

            
        self.colormap_type = colormap_type
                       
        self.show_bounding_box = bounding_box
        
        self.transform_type = transform_type

        self.view_type = view_type

        self.camera_type = camera_type
                       
        # Initialize constants for the system
        
        # set to some default values if None is passed in
        self.azimuth = 0 if azimuth is None else azimuth
        self.elevation = 0 if elevation is None else elevation
        #self.roll = 0 if roll is None else roll
        self.x_translate = 0 if x_translate is None else x_translate
        self.y_translate = 0 if y_translate is None else y_translate
        self.z_translate = 0 if z_translate is None else z_translate
        
        self.scale = 1200.0 if scale is None else scale
        self.fov_angle_x =  94 if fov_angle_x is None else fov_angle_x
        self.fov_angle_y =  fov_angle_y if fov_angle_y is None else fov_angle_y
        self.range_max = None if range_max is None else range_max
        self.range_min = None if range_min is None else range_min
        self.ampl_min = self.ampl_min if ampl_min is None else ampl_min
        self.x_max = None if x_max is None else x_max
        self.x_min = None if x_min is None else x_min
        self.y_max = None if y_max is None else y_max
        self.y_min = None if y_min is None else y_min
        self.reflectivity_thresh = reflectivity_thresh#0 if reflectivity_thresh is None else reflectivity_thresh
        self.colorize = colorize
        self.qp_ampl = qp_ampl
        self.qp_phase = qp_phase
        # model view controller signal slot
        
        
    def set_data(self,data,imaging_type):
        '''
        data is update from the camera here and then set for plotting
        '''
        start_of_update = time.time()
        filtered_phase = None
        if data is not None:
            
            pre_time = data[0]
            post_time = data[1]
            camera_data = data[2]
            camera_index = int(data[3])-1 #hack
            
            
            cnt  = 0
            ind = 0
            frame = camera_data[ind][0]
            camera_chip_ip = camera_data[ind][1]
            frame_number = camera_data[ind][2][0][0]
            camera_imaging_type = camera_data[ind][3]
            current_frame = frame_number
            assert imaging_type[ind] == camera_imaging_type
            #print('View update  ',np.mean(camera_data[0][0]),     time.time(),end='')
            print('View update frame_number %d chip %d  '%(frame_number,camera_index))
            # ToDo
            # given value of server IP (camera chip) look up key and convert to integer and make it [0...]
            # hacked above for testing
            # camera_index = next(key for key, value in self.chips.items() if value == camera_chip)
            # camera_index = int(camera_index)-1
                
            
            if imaging_type[ind] == 'Dist_Ampl':
            
                #####to scale and threshold#######################################################################
                #print("sum,max before ", np.sum(to_display[ind]))
                # the reason for below is that ESPROS does it this way; even if there is binning the data is interpolated to standard (240,320) size before pcl
                # if  self.interpolation :
                #     _f0 = cv2.resize(frame[:,:,0],(240,320), interpolation = cv2.INTER_NEAREST)
                #     _f1 = cv2.resize(frame[:,:,1],(240,320), interpolation = cv2.INTER_NEAREST)
                #     del frame
                #     frame = np.zeros((320,240,2))
                #     frame[:,:,0] = _f0
                #     frame[:,:,1] = _f1
               
                
                #print( np.where(np.squeeze(frame[:,:,0])>30000)[0].shape)
                raw_ampl = np.squeeze(frame[:,:,1])
                #amplitude_saturated_indices = np.where(raw_ampl>=self.AMPLITUDE_SATURATION)
                #print("Amplitude Saturated indices =", amplitude_saturated_indices[0].shape)
                #if self.saturation_flag:
                #    frame[:,:,0][np.where(ampl==self.AMPLITUDE_SATURATION)] = 0
                #if self.adc_flag:
                #    frame[:,:,0][np.where(ampl==self.ADC)] = 0 
                # amplitude_low_indices = np.where(raw_ampl==self.LOW_AMPLITUDE)
                # amplitude_beyond_range_indices = np.where(np.logical_or(raw_ampl < self.ampl_min,
                #                                                             np.logical_and(raw_ampl>self.MAX_AMPLITUDE,raw_ampl<self.LOW_AMPLITUDE)))
                #print("Low Amplitude Indices =", low_amplitude_indices[0].shape)
                
                raw_phase = np.squeeze(frame)[:,:,0]

                rgb_img = None
                if self.rgb and data[4] is not None:
                    rgb_img = data[4]

                rgb_img,raw_phase,raw_ampl = camera_calibrations(rgb_img,depth=raw_phase,ampl=raw_ampl,camera_version = self.camera_version)

                    
                #self.got_image = True
                #print(rgb_img.shape,raw_phase.shape,self.number_chips)

                #if self.camera_version=='oyla_1_camera_v0':
                    #this should not be needed
                #    raw_ampl = np.fliplr(raw_ampl)
                #    raw_phase = np.fliplr(raw_phase)
                    

                # phase_min = self.range_min*self.MAX_PHASE/self.ambiguity_distance
                # phase_max = self.range_max*self.MAX_PHASE/self.ambiguity_distance
                # phase_beyond_range_indices = np.where(np.logical_and(np.logical_or(raw_phase<phase_min, raw_phase>phase_max),raw_phase>0))
                if 'temporal_median_filter' in self.filter_params and int(self.filter_params['temporal_median_filter']):
                    print('Doing temporal median filter',frame_number)
                    frame_offset = int(self.filter_params['temporal_median_filter_size'])
                    #frame_offset = 8
                    if frame_number == 0:                            
                        self.median_array = np.zeros((frame_offset+1+frame_offset, raw_phase.shape[0], raw_phase.shape[1],2))
                        self.rgb_array = np.zeros((frame_offset+1+frame_offset, rgb_img.shape[0], rgb_img.shape[1],3),dtype = np.uint8)
                    if frame_number < frame_offset+frame_offset+1:

                        self.median_array[frame_number, :, :, 0] = raw_phase
                        self.median_array[frame_number, :, :, 1] = raw_ampl
                        self.rgb_array[frame_number,:,:,:] = rgb_img
                    else:
                        current_frame = frame_number - frame_offset


                    if current_frame >=frame_offset and (current_frame+frame_offset+1) < self.total_num_of_images:
                        if current_frame > frame_offset:
                            self.median_array = np.roll(self.median_array, -1, axis=0)
                            self.rgb_array = np.roll(self.rgb_array, -1, axis = 0)
                            self.median_array[frame_offset+frame_offset+0, :, :, 0] = raw_phase
                            self.median_array[frame_offset+frame_offset+0, :, :, 1] = raw_ampl
                            self.rgb_array[frame_offset+frame_offset+0, :, :, :] = rgb_img
                        #slice_list = []
                        #if (1+current_frame) > self.number_images:
                        #    break
                        #else:
                        #        slice_list.append(median_array[0:(0+frame_offset+frame_offset)])
                        results = filter_temporal_median(self.median_array[0:(0+frame_offset+frame_offset+1)])
                        raw_phase = results[0]
                        raw_ampl = results[1]
                        rgb_img = self.rgb_array[frame_offset,:,:,:]
                   
                filtered_phase, thresholded_ampl, indices = threshold_filter(raw_phase = raw_phase, raw_ampl = raw_ampl, reflectivity_thresh= self.reflectivity_thresh,
                                                                             range_max = self.range_max, range_min = self.range_min, ampl_min = self.ampl_min,
                                                                             filter_params = self.filter_params, ambiguity_distance = self.ambiguity_distance,
                                                                             qp_phase = self.qp_phase, qp_ampl = self.qp_ampl)
                
                #filtered_phase = img #
                
                dist, _ = phase_to_distance(filtered_phase,self.ambiguity_distance)
                
                #print(no_data_indices[0].shape)
                # no_data_indices = (np.concatenate((no_data_indices[0],amplitude_saturated_indices[0])),
                #                          np.concatenate((no_data_indices[1],amplitude_saturated_indices[1])))
                #print(no_data_indices[0].shape)
                # No conversion The problem of using convert matrix below is that it makes it a RGBA data,
                # which gives error in point cloud plotting. 
                # depth_image_array = convert_matrix_image(img,cmap= 'jet_r', clim_min=self.range_min, clim_max=self.range_max,
                #                            saturation_indices = amplitude_saturated_indices,
                #                            no_data_indices = low_amplitude_indices)
                #also we are not yet removing outside range -- why did we not do this?

                if self.rgb and rgb_img is not None:
                    if self.camera_version == 'oyla_1_camera_v3_cds':
                        _rgb_img = rgb_img
                    else:
                        _rgb_img = cv2.resize(rgb_img, None,fx=0.65,fy=0.65)###(320,120)) #scaling is related to view_rgb size set above
                    #     d,_,_ = rgb_depth_view_matching(data[4],self.number_chips)
                    #     rgb_img = cv2.resize(d, None,fx=0.75,fy=0.75)###(320,120)) #scaling is related to view_rgb size set above
                    #if self.camera_version == 'oyla_1_camera_v0':
                    #    _rgb_img = np.fliplr(_rgb_img)
                    self.rgb_image.set_data(_rgb_img) #this should not be needed
                    self.got_image = True
            elif imaging_type[ind] == 'HDR':
                raw_phase = np.squeeze(frame)[:,:,3]
                raw_ampl = np.squeeze(frame)[:,:,4]
                # this uses amplitude to deinterlace distance (img)
                raw_ampl, raw_phase = getComparedHDRAmplitudeImage_vectorized(raw_ampl, raw_phase)
                #amplitude_saturated_indices = np.where(np.squeeze(raw_ampl>=self.AMPLITUDE_SATURATION))
                #raw_phase = img.copy()
                # amplitude_low_indices = np.where(raw_ampl == self.LOW_AMPLITUDE)
                # amplitude_beyond_range_indices = np.where(np.logical_or(raw_ampl < self.ampl_min,
                #                                                         np.logical_and(raw_ampl>self.MAX_AMPLITUDE,raw_ampl<self.LOW_AMPLITUDE)))
                    
                # phase_min = self.range_min*self.MAX_PHASE/self.ambiguity_distance
                # phase_max = self.range_max*self.MAX_PHASE/self.ambiguity_distance
                # phase_beyond_range_indices = np.where(np.logical_and(np.logical_or(raw_phase<phase_min, raw_phase>phase_max),raw_phase>0))
                filtered_phase, thresholded_ampl, indices = threshold_filter(raw_phase = raw_phase, raw_ampl = raw_ampl, reflectivity_thresh= self.reflectivity_thresh,
                                                                             range_max = self.range_max, range_min = self.range_min, ampl_min = self.ampl_min,
                                                                             filter_params = self.filter_params, ambiguity_distance = self.ambiguity_distance)    
                
                dist, _ = phase_to_distance(filtered_phase,self.ambiguity_distance)
                
            # All data saturated and low amplitude has to be ignored along with data that is either outside range or reflectivity 
            #print(outside_range_indices[0].shape)
            no_data_indices = indices['amplitude_saturated']
            for k in indices.keys():
                if k != 'amplitude_saturated':
                    no_data_indices = (np.concatenate((no_data_indices[0], indices[k][0])),
                                       np.concatenate((no_data_indices[1], indices[k][1])))
                    
            # no_data_indices = (np.concatenate((amplitude_saturated_indices[0],amplitude_low_indices[0])),
            #                    np.concatenate((amplitude_saturated_indices[1],amplitude_low_indices[1])))
            # no_data_indices = (np.concatenate((no_data_indices[0], phase_beyond_range_indices[0])),
            #                    np.concatenate((no_data_indices[1], phase_beyond_range_indices[1])))
            # no_data_indices = (np.concatenate((no_data_indices[0], amplitude_beyond_range_indices[0])),
            #                    np.concatenate((no_data_indices[1], amplitude_beyond_range_indices[1])))
            # no_data_indices = (np.concatenate((no_data_indices[0], filtered_phase_beyond_range_indices[0])),
            #                    np.concatenate((no_data_indices[1], filtered_phase_beyond_range_indices[1])))
            # if self.reflectivity_thresh:
            #     no_data_indices = (np.concatenate((no_data_indices[0], reflectivity_indices[0])),
            #                        np.concatenate((no_data_indices[1], reflectivity_indices[1])))
           
            # we still have an array of depth numbers and some indices -- this is fed into transformation to cartesian; also note no_data_indices will remove points in pcl
            # which have either low amplitude or are saturated -- here we go from full to sparse 
            y, x, z, rcm = transformation3(dist,self.transform_type,self.fov_angle_x,self.fov_angle_y,
                                           no_data_indices)
            
            #np_points_array = np.zeros((len(x),3))                
            #np_points_array[:,0], np_points_array[:,1], np_points_array[:,2] = x,y,z
            #print(np.mean(np_points_array,axis=0),np.max(np_points_array,axis=0),np.min(np_points_array,axis=0))
            # this function will threshold each coordinate based on range selected by UI

            x, y, z, rcm = threshold_coordinates(x, y, z, rcm, x_max = self.x_max, x_min = self.x_min,
                                                 y_max = self.y_max, y_min = self.y_min)
            # ToDo: why is this needed?
            np_points_array = np.zeros((len(x),3))                
            np_points_array[:,0], np_points_array[:,1], np_points_array[:,2] = x,y,z
            
            #print(np.mean(np_points_array,axis=0))
            #self.show_bounding_box = False
            # if not self.show_bounding_box:
            #     _a = np_points_array
            #     _a = np.c_[_a,rcm]#np.zeros(np.shape(_a)[0])] 
            #     _a = _a.astype('single')
            #     _a.tofile('oyla_'+str(frame_number)+'.bin')

            # color matrix for each point in pointcloud based on range value
            rcm -= np.min(self.range_min)
            rcm /= np.max(self.range_max)
            np_color_array = cm.get_cmap('jet_r')(rcm)

            #self.canvas.cbar_widget.cmap = colormap
            self.cbar.clim = (self.range_min, self.range_max)
            self.view.camera = self.camera_type#'turntable'  ## or try 'arcball' turntable  'panscale',
            #print('xxx',self.view.camera.roll,self.roll)
            # what view we are looking at
            if(self.view_type == 'bird_eye_view'): # 
                self.view.camera.azimuth = 0
                self.view.camera.elevation = 0
            elif(self.view_type == 'camera_view'): # 0 -90
                self.view.camera.azimuth = 0
                self.view.camera.elevation = -90
            elif self.view_type == 'other_axis_view':
                self.view.camera.azimuth = 90
                self.view.camera.elevation = 0
            else:
                self.view.camera.azimuth = self.azimuth
                self.view.camera.elevation = self.elevation
                self.view.camera.roll = 0
            print("number of points",np_points_array.shape)
            # actuall plot of 3 d dtaa along with color data
            self.scatter.set_data(np_points_array,edge_color= np_color_array, face_color= np_color_array, size=1.5)
            self.view.add(self.scatter)
            if self.txt_to_be_displayed is not None:
                
                try:
                   self.textVis.text = str(round(float(self.txt_to_be_displayed[str(frame_number[0][0])])/100,2))
                  
                except:
                    print(self.txt_to_be_displayed,frame_number[0][0])
                    
            self.view.camera.center = [self.x_translate, self.y_translate,
                                       self.z_translate] 
            self.view.camera.scale_factor = self.scale

            end_of_update = time.time()

            if(self._closed):
                app.quit()
            self.update()
                #print("Total time taken for one update in milli sec", 1000*(end_of_update - start_of_update))
        
        return filtered_phase, thresholded_ampl,rgb_img
                  
                  
                  
                   
