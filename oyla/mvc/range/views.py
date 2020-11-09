##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Creation:          20.07.2019                                                                  #
# Version:           1.0                                                                         #
# Revision history:  Initial version from vispy examples (Isoline) codebase
# Version: 2.0, 20.10.2019
# RGB, filtereing are major changes, filtering only for Dist_Ampl and HDR mode
# Description                                                                                    #
#        sets up the window -- QtWidget
##################################################################################################


import sys
import numpy as np
import time
from vispy import scene
import cv2
from vispy.geometry.generation import create_sphere
from vispy.color.colormap import get_colormaps
import vispy.plot as vp
try:
    from sip import setapi
    setapi("QVariant", 2)
    setapi("QString", 2)
except ImportError:
    pass


from PyQt5.QtCore import pyqtSignal, Qt, QThread, pyqtSlot
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                                 QSpinBox, QComboBox, QGridLayout, QVBoxLayout,
                                 QSplitter)
from oyla.utils import convert_matrix_image
from oyla.mvc.utils import getComparedHDRAmplitudeImage_vectorized, phase_to_distance, camera_calibrations, threshold_filter
from oyla.mvc.utils import AMPLITUDE_SATURATION, GRAYSCALE_SATURATION, LOW_AMPLITUDE, MAX_AMPLITUDE
from oyla.mvc.filters import filter_temporal_median
        
        
class Canvas(scene.SceneCanvas):
    '''
    This is where the plotting will happen
    set an max_number_images (could use less at run time)
    set which of the images should show a colorbar
    also send the chips information, this is the IP we setup; we can use the same to match chip information in data returned from camera
    slight convolution because teh returned data will have IP as identifier not chip number
    '''
    def __init__(self, azimuth = None, elevation = None, roll = None, x_translate=None, y_translate=None, z_translate=None,
                 scale=None, bounding_box=False, colormap_type = 'rgb', view_type = 'default_view',
                 transform_type ='cartesian', camera_type = 'turntable',fov_angle_x = None, fov_angle_y = None, setup_widget = None,
                 range_max = None, range_min = None, x_max = None, x_min = None, y_max= None, y_min = None,rgb = False,ampl_min = 50, reflectivity_thresh = None,
                 max_number_images=3, chips = None,show_colorbar = [], colorize = False, qp_phase = 1, qp_ampl = 1):
        # For vispy this is the man drawing class
        # parameters such that there can be max_number of images/views
        
        scene.SceneCanvas.__init__(self, keys=None)
        self.size = 1300*1.0, 700*1.0

        # Unfreeze to creates nodes of a graph that are then rendered at run time
        self.unfreeze()
        self.chips = chips
        self.show_colorbar = show_colorbar
        self.grid = self.central_widget.add_grid(margin=1)
        self.setup_widget  = setup_widget
        self.set_parameters( azimuth = azimuth, elevation = elevation,roll = roll,
                             x_translate=x_translate, y_translate=y_translate,
                             z_translate=z_translate,scale=scale, bounding_box=bounding_box,
                             colormap_type = colormap_type, view_type = view_type,
                             transform_type = transform_type, fov_angle_x = fov_angle_x, fov_angle_y = fov_angle_y,
                             camera_type = camera_type,
                             range_max = range_max, range_min = range_min, x_max = x_max, x_min = x_min,
                             y_max= y_max, y_min = y_min, ampl_min = ampl_min, reflectivity_thresh = reflectivity_thresh, colorize = colorize,
                             qp_ampl = qp_ampl, qp_phase = qp_phase)
        ###########################################################
        # this is the tricky part that we are initializing more "vbs" than needed to allow for multiple epochs
        # Canvas is initialized in Main Window before it starts up
        ##########################################################
        self.vbs = []
        self.images = []
        self.colorbars = []
        for image_number in range(max_number_images):
            
            self.colorbars.append(scene.ColorBarWidget(label="", 
                                                       cmap='grays', orientation="left", border_width=1,
                                                       border_color = "#212121",label_color="white"))
            # condition that either gets a vbs as a view box, or a vbs as a widget that has view box and colorbar
            # set color bar so that it shows correctly for what ever images you want
            # more colorbars than needed are initialized (above) only to make indexing easy
            if image_number not in self.show_colorbar:
                self.vbs.append(scene.widgets.ViewBox(border_color='red', parent=self.scene))
            else:
                _widget = scene.widgets.ViewBox(border_color='red', parent=self.scene)
            
                _grid = _widget.add_grid()
                _grid.add_widget(self.colorbars[-1], col = 25)
                self.vbs.append(_widget)
                
            self.images.append(scene.visuals.Image(parent = self.vbs[-1].scene,picking=True))
            self.images[-1].cmap = 'viridis'

        # initially the grid has only one view box
        # all this is placeholder; assumption is update_canvas will update all values before viewing
        self.grid.add_widget(self.vbs[0],0,0)
        self.number_images = 1
        self.updated_canvas = True
        self.got_image = False
        self.number_chips = 1
        
        self.ambiguity_distance = None
        self.range_max = None
        self.range_min = None
        self.ampl_min = None
        self.saturation_flag = False
        self.adc_flag = False
        self.mod_freq = None
        self.filter_params = {}
        self.rgb = False
        self.view_as = 'range'
        self.reflectivity_thresh = 0
        self.colorize = False
        self.camera_version = 'oyla_2_camera'
        self.filtered_phase = None
        self.thresholded_ampl = None
        self.cursor_text = vp.Text("", pos=(0, 0), anchor_x='left', anchor_y='center', color='red',
                      font_size=8, parent=self.scene)
        self.show_text = False
        self.median_array = None
        self.rgb_array = None
        self.total_num_of_images = 0
        #self.cursor_text.order = 10
        # all members of canvas class have to be initialized here where canvas is unfrozen
        
              
        #canvas is now frozen that is view boxes, colorbars etc can not be added. Image data can be set to view box
        self.freeze()

    def on_mouse_press(self,event):
        self.show_text = not self.show_text
        self.cursor_text.text = ""
    def on_mouse_move(self,event):
        if self.show_text:
            #print("doing nothing on mouse press ---------------------")
            #print(self.view.camera.get_state())
            #print(event.pos)

            #for i in range(self.number_images):
                #print(self.vbs[i].get_scene_bounds(),self.vbs[i].pos,self.vbs[i].inner_rect)
            print(event.pos,self.size)
            pos = self.vbs[0].scene.transform.imap(event.pos)
            pos = list(map(int,pos))
            sb = self.vbs[0].get_scene_bounds()
            pos[1] = sb[1][1]-pos[1]
            if pos[0]<sb[0][1] and pos[1]<sb[1][1] and pos[0]>0 and pos[1]>0:
                x = event.pos[0]
                y = event.pos[1]
                self.cursor_text.text = "phase=%0.2f, ampl=%0.2f" % (self.filtered_phase[pos[1],pos[0]], self.thresholded_ampl[pos[1],pos[0]])
                offset = np.diff(self.vbs[0].scene.transform.map([[0, 0], [0,0]]), axis=0)[0, 0]
                self.cursor_text.pos = self.size[0]/2+10, self.size[1]/2+10
        
    def update_canvas(self,number_images, number_chips = 1, ambiguity_distance = None,range_max = 9 ,range_min = 0.1, saturation_flag = None ,
                      adc_flag = None,mod_freq = None, filter_params = {},rgb = False, view_as = 'range', ampl_min = 50, reflectivity_thresh = False,
                      colorize=False, camera_version = 'oyla_2_camera', total_num_of_images = 0):
        ######################################################################
        # this allows for canvas to be updated after each epoch. It reinitializes the grid
        # Full control on how many grid widgets and which vbs to in which widget
        # no control over vbs itself
        #  image.set_data(None) does not work -- but is convinient for now
        # can reinitialize plotting parameters here 
        #######################################################################
        for i in range(self.number_images):
            self.vbs[i].border_color = None
            self.vbs[i].camera.reset()
            self.grid.remove_widget(self.vbs[i])
            self.images[i].set_data(None)
        
        self.number_images = number_images
        self.number_chips = number_chips
        
        for i in range(self.number_images):
            self.grid.add_widget(self.vbs[i],i%2,i//2)
        self.updated_canvas = True*np.ones(self.number_images)
        # this is used to identify that a view box has got image for the first time
        self.got_image = False*np.ones(self.number_images)
        
        self.ambiguity_distance = ambiguity_distance
        self.range_max = range_max
        self.range_min = range_min
        self.setup_widget.psets[8].setValue(range_max)
        self.setup_widget.psets[9].setValue(range_min)
        self.setup_widget.psets[10].setValue(ampl_min)
        self.saturation_flag = saturation_flag
        self.adc_flag = adc_flag
        self.mod_freq = mod_freq
        self.filter_params = filter_params
        self.rgb = rgb
        self.view_as = view_as
        self.reflectivity_thresh = reflectivity_thresh
        self.setup_widget.reflectivity_thresh_chk.setChecked(reflectivity_thresh)
        self.colorize = colorize
        self.setup_widget.colorize_chk.setChecked(colorize)
        self.ampl_min =  ampl_min
        self.camera_version = camera_version
        self.total_num_of_images = total_num_of_images
        # if using properties widget
        #def set_data(self, n_images, cmap,to_display=None,imaging_type = []):
        #self.iso.set_color(cmap)
        #cl = np.linspace(-self.radius, self.radius, n_levels + 2)[1:-1]
        #self.iso.levels = cl
    def set_parameters(self, azimuth = None, elevation = None, roll = None, x_translate=None,y_translate=None,z_translate=None,
                       scale=None, bounding_box=False, colormap_type = 'grays', view_type = 'default_view',
                       transform_type = 'cartesian', fov_angle_x = 94, fov_angle_y = 94,camera_type = 'turntable',
                       range_max = None, range_min = None, x_max = None, x_min = None, y_max= None, y_min = None, ampl_min = None,
                       reflectivity_thresh = False,colorize = False, qp_phase = 1, qp_ampl = 1):
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
        self.roll = 0 if roll is None else roll
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
        self.reflectivity_thresh = reflectivity_thresh#None if reflectivity_thresh is None else reflectivity_thresh
        self.colorize = colorize
        self.qp_phase = qp_phase
        self.qp_ampl = qp_ampl
        # model view controller signal slot

    def set_data(self, data=None,imaging_type = []):
        
        #################################################################
        # main function for updating view_boxes by setting data of image child of view box
        # data is from camera and has pre (when camera was send command) post (when camera responded with data) times
        # and camera_data which is data coming from epc_image and is list
        # [image, serverIP/cjip, frame_number, camera_imaging_type]
        # camera_imaging_type will differ from imaging_type in that latter is a list in case of X+Y modalities
        #################################################################
        if data is not None:
            #print(camera_data[2])
            pre_time = data[0]
            post_time = data[1]
            camera_data = data[2]
            camera_index = int(data[3])-1 #hack that works only because even in case of multiple cameras we are basically using camera '1' as identifier after stitching
            
            #print('View update at ',time.time(),np.mean(camera_data[0][0]),end=' ')
                  #camera_data[0][0].shape,self.vbs[0].camera.get_state(),imaging_type)
            
            cnt  = 0
            filtered_phase = None
            thresholded_ampl = None
            rgb_img = None
            for ind in range(len(imaging_type)):
                
                frame = camera_data[ind][0]
                camera_chip_ip = camera_data[ind][1]
                frame_number = camera_data[ind][2][0][0]
                camera_imaging_type = camera_data[ind][3]
                current_frame = frame_number
                assert imaging_type[ind] == camera_imaging_type
                #given value of server IP (camera chip) look up key and convert to integer and make it [0...]
                # hacked above for testing
                #camera_index = next(key for key, value in self.chips.items() if value == camera_chip)
                #camera_index = int(camera_index)-1
                
                print('View update frame_number %d chip %d  '%(frame_number,camera_index))
                if imaging_type[ind] == 'Gray':
                    if self.view_as is not None:
                        grayscale_saturated_indices = np.where(np.squeeze(frame>=GRAYSCALE_SATURATION))
                        #print("Grayscale Saturated indices =", np.count_nonzero(grayscale_saturated_indices))
                        img = np.squeeze(frame)
                        img = convert_matrix_image(img,cmap= 'gray', clim_min=2048, clim_max=4095,
                                             saturation_indices = grayscale_saturated_indices)
                        cnt = self.plot(img,cnt,camera_index,'gray')
                elif imaging_type[ind] == 'Dist_Ampl':
            
                    #####to scale and threshold#######################################################################
                    #print("sum,max before ", np.sum(to_display[ind]))
                    # below takes care of ADC also
                    raw_ampl = np.squeeze(frame[:,:,1])
                    #amplitude_saturated_indices = np.where(raw_ampl>=self.AMPLITUDE_SATURATION)
                    #print("Amplitude Saturated indices =", np.count_nonzero(amplitude_saturated_indices))
                    #In earlier versions we had saturation flag and adc flag and amplitude_min which we have remoed
                   
                    # amplitude_low_indices = np.where(raw_ampl == self.LOW_AMPLITUDE)
                    
                    # amplitude_beyond_range_indices = np.where(np.logical_or(raw_ampl < self.ampl_min,
                    #                                                         np.logical_and(raw_ampl>self.MAX_AMPLITUDE,raw_ampl<self.LOW_AMPLITUDE)))
                    #print("Low Amplitude Indices =", amplitude_low_indices[0].shape,amplitude_beyond_range_indices[0].shape)

                    
                    raw_phase = np.squeeze(frame)[:,:,0]
                    # phase_min = self.range_min*self.MAX_PHASE/self.ambiguity_distance
                    # phase_max = self.range_max*self.MAX_PHASE/self.ambiguity_distance
                    # phase_beyond_range_indices = np.where(np.logical_and(np.logical_or(raw_phase<phase_min, raw_phase>phase_max),raw_phase>0))

                    # if you need to hack the raw phase to be displayed along with filtered_phase
                    #raw_phase = img.copy()
                    #threshold raw phase and raw amplitude
                    # moved up so that we can upsample and crop depth, amplitude imaegs
                    rgb_img = None
                    if self.rgb and data[4] is not None:
                        rgb_img = data[4]
                    #rgb_img,raw_phase,raw_ampl = camera_calibrations(rgb=rgb_img,depth=raw_phase,ampl=raw_ampl,camera_version = self.camera_version)
                    if 'temporal_median_filter' in self.filter_params and int(self.filter_params['temporal_median_filter']):
                        print('Doing temporal median filter',frame_number)
                        frame_offset = int(self.filter_params['temporal_median_filter_size'])
                        if frame_number == 0:                            
                            self.median_array = np.zeros((frame_offset+1+frame_offset, raw_phase.shape[0], raw_phase.shape[1],2))
                            self.rgb_array = np.zeros((frame_offset+1+frame_offset, rgb_img.shape[0], rgb_img.shape[1],3),dtype = np.uint8)
                        if frame_number < frame_offset+frame_offset+1:
                            
                            self.median_array[frame_number, :, :, 0] = raw_phase
                            self.median_array[frame_number, :, :, 1] = raw_ampl
                            self.rgb_array[frame_number,:,:,:] = rgb_img
                            print('far',frame_number)
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
                            #if (1+current_frame) > self.total_num_of_images:
                            #    break
                            #else:
                            #        slice_list.append(median_array[0:(0+frame_offset+frame_offset)])
                            results = filter_temporal_median(self.median_array[0:(0+frame_offset+frame_offset+1)])
                            raw_phase = results[0]
                            raw_ampl = results[1]
                            rgb_img = self.rgb_array[frame_offset,:,:,:]
                            print('Now filter',frame_offset,np.mean(rgb_img))
                    filtered_phase, thresholded_ampl, indices = threshold_filter(raw_phase = raw_phase, raw_ampl = raw_ampl, reflectivity_thresh= self.reflectivity_thresh,
                                                                             range_max = self.range_max, range_min = self.range_min, ampl_min = self.ampl_min,
                                                                                 filter_params = self.filter_params, ambiguity_distance = self.ambiguity_distance,
                                                                                 qp_phase = self.qp_phase, qp_ampl = self.qp_ampl)

                    #rgb_img,filtered_phase,thresholded_ampl = camera_calibrations(rgb=rgb_img,depth=filtered_phase,ampl=thresholded_ampl,camera_version = self.camera_version)
                    self.filtered_phase = filtered_phase
                    self.thresholded_ampl = thresholded_ampl
                    # here distance is radial distance i.e. range not depth, distance from plane. hence use of range_max, range_min is correct
                    if self.view_as is not None:
                        # no plotting if view_as is None; the range_max, range_min have been removed as arguments as they have already been thresholded above
                        dist,outside_range_indices = phase_to_distance(filtered_phase, self.ambiguity_distance)#, range_max = self.range_max, range_min = self.range_min)

                        #_d = np.zeros_like(dist,dtype=np.uint16)
                        #_d = np.uint16(dist/256)*256
                        #dist = _d.astype('float32')
                        #assert outside_range_indices==phase_beyond_range_indices, #this assertion will fail as we have already thresholded outside range values
                        
                        outside_range_indices = indices['amplitude_beyond_range']
                        for k in indices.keys():
                            if 'beyond_range' in k and k != 'amplitude_beyond_range':
                                outside_range_indices = (np.concatenate((outside_range_indices[0], indices[k][0])),
                                                         np.concatenate((outside_range_indices[1], indices[k][1])))
                        # outside_range_indices = (np.concatenate((amplitude_beyond_range_indices[0], phase_beyond_range_indices[0])),
                        #                         np.concatenate((amplitude_beyond_range_indices[1],phase_beyond_range_indices[1])))
                        # outside_range_indices = (np.concatenate((outside_range_indices[0], filtered_phase_beyond_range_indices[0])),
                        #                         np.concatenate((outside_range_indices[1], filtered_phase_beyond_range_indices[1])))
                        #if self.reflectivity_thresh:
                        #    outside_range_indices = (np.concatenate((outside_range_indices[0], reflectivity_indices[0])),
                        #                             np.concatenate((outside_range_indices[1], reflectivity_indices[1])))
                            

                        # above phase is converted to distance, here for index in indices values are set to predefined constants and colormap is imposed
                            
                        img = convert_matrix_image(dist,cmap= 'jet_r', clim_min=self.range_min, clim_max=self.range_max,
                                                   saturation_indices = indices['amplitude_saturated'],
                                                   no_data_indices = indices['amplitude_low'],
                                                   outside_range_indices = outside_range_indices, colorize = self.colorize)
                        fp = img
                        # actual plotting
                        #cnt = self.plot(img,cnt,camera_index,'jet_r',clim=(self.range_min,self.range_max))
                        # now plotting ampl note hard coding of clim_max
                        
                        img = convert_matrix_image(thresholded_ampl,cmap= 'jet_r', clim_min=self.ampl_min, clim_max=MAX_AMPLITUDE/4,
                                                   saturation_indices = indices['amplitude_saturated'],
                                                   no_data_indices = indices['amplitude_low'],
                                                   outside_range_indices = outside_range_indices, colorize = self.colorize)
                        ampl = img
                        rgb_img,fp,ampl = camera_calibrations(rgb=rgb_img,depth=fp,ampl=ampl,camera_version = self.camera_version)
                        print(cnt, fp.shape)
                        cnt = self.plot(fp,cnt,camera_index,'jet_r',clim=(self.range_min,self.range_max))
                        cnt = self.plot(ampl,cnt,camera_index,'jet_r',clim=(self.ampl_min,MAX_AMPLITUDE/4))
                        # Can be used to add original plot in case of filtered data
                        # rgb_img = None
                        # img,outside_range_indices = phase_to_distance(raw_phase, self.ambiguity_distance, range_max = self.range_max,
                        #                                                                   range_min = self.range_min)
                        # img = convert_matrix_image(img,cmap= 'jet_r', clim_min=self.range_min, clim_max=self.range_max,
                        #                            saturation_indices = amplitude_saturated_indices,
                        #                            no_data_indices = low_amplitude_indices,
                        #                            outside_range_indices = outside_range_indices)
                        # cnt = self.plot(img,cnt,camera_index,'jet_r',clim=(self.range_min,self.range_max))
                        if self.rgb and rgb_img is not None:
                            # # this was needed here because in plot all data gets rotated I am going to get all the flip etc here
                            # #_rgb_img = np.transpose(rgb_img,(1,0,2))
                            # if self.camera_version == 'oyla_1_camera_v0':
                            #     #pure hack also because saving may not have been correct
                            #     _rgb_img = np.flipud(np.fliplr(_rgb_img))
                            cnt = self.plot(rgb_img,cnt,camera_index)
                elif imaging_type[ind] == 'Dist':
                    # only convert phase to distance, map values for some indices, and impose colormap
                    if self.view_as is not None:
                        img = np.squeeze(frame)
                        img,outside_range_indices = phase_to_distance(img,self.ambiguity_distance, range_max = self.range_max,
                                                                      range_min = self.range_min)
                        distance_saturated_indices = np.where(np.squeeze(frame)==AMPLITUDE_SATURATION)
                        img = np.squeeze(frame)
                        img = convert_matrix_image(img,cmap= 'jet_r', clim_min=0.5, clim_max=5,
                                                   saturation_indices = distance_saturated_indices,
                                                   outside_range_indices = outside_range_indices,colorize = self.colorize)
                        cnt = self.plot(img,cnt,camera_index,'jet_r')

                elif imaging_type[ind] == 'Ampl':
                    # only convert phase to distance, map values for some indices, and impose colormap
                    if self.view_as is not None:
                        frame[np.where(frame==LOW_AMPLITUDE)] = 0
                        img = np.squeeze(frame)
                        distance_saturated_indices = np.where(np.squeeze(frame)==AMPLITUDE_SATURATION)
                        img = convert_matrix_image(img,cmap= 'gnuplot_r', clim_min=1, clim_max=2000,
                                                   saturation_indices = distance_saturated_indices,colorize = self.colorize)
                        cnt = self.plot(img,cnt,camera_index,'gnuplot_r')
                    
                elif imaging_type[ind] == 'DCS' :
                    # only convert phase to distance, map values for some indices, and impose colormap
                    # note clim is set by median of each frame

                    if self.view_as is not None:
                        raw0  = np.squeeze(frame)[:,:,0]
                        raw1 = np.squeeze(frame)[:,:,1]
                        raw2 = np.squeeze(frame)[:,:,2]
                        raw3 = np.squeeze(frame)[:,:,0] + np.squeeze(frame)[:,:,2]
                        rgb_img = None
                        if self.rgb and data[4] is not None:
                            rgb_img = data[4]
                        rgb_img,raw0,raw1 = camera_calibrations(rgb=rgb_img,depth=raw0,ampl=raw1,camera_version = self.camera_version)
                        _,raw2,raw3 = camera_calibrations(rgb=None,depth=raw0,ampl=raw1,camera_version = self.camera_version)
                        median_scale =  np.median(raw0)
                        img = convert_matrix_image(raw0,cmap= 'viridis', 
                                                   clim_min=median_scale-500, clim_max=median_scale+500)
                        cnt = self.plot(img,cnt,camera_index,'viridis')
                        
                        img = convert_matrix_image(raw1,cmap= 'viridis', 
                                                   clim_min=median_scale-500, clim_max=median_scale+500)
                        cnt = self.plot(img,cnt,camera_index,'viridis')
                        
                        img = convert_matrix_image(raw2,cmap= 'viridis', 
                                                   clim_min=median_scale-500, clim_max=median_scale+500)
                        cnt = self.plot(img,cnt,camera_index,'viridis')
                        
                        img = convert_matrix_image(raw3,cmap= 'viridis', 
                                                   clim_min=2*median_scale-200, clim_max=2*median_scale+200)
                        cnt = self.plot(img,cnt,camera_index,'viridis')
                        
                        if self.rgb and rgb_img is not None:
                            # this was needed here because in plot all data gets rotated I am going to get all the flip etc here
                            #_rgb_img = np.transpose(rgb_img,(1,0,2))
                            # if self.camera_version == 'oyla_1_camera_v0':
                            #     #pure hack also because saving may not have been correct
                            #     _rgb_img = np.flipud(np.fliplr(_rgb_img))
                            cnt = self.plot(_rgb_img,cnt,camera_index)
                elif imaging_type[ind] == 'HDR':
                    # 0,1,2 HDR are plotted as is convert phase to distance, map values for some indices, and impose colormap
                    # 2 is converted to distance using amplitude (3)
                    # distance and amplitude can be filtered
                    if self.view_as is not None:
                        img = np.squeeze(frame)[:,:,0]
                        amplitude_saturated_indices = np.where(np.squeeze(frame[:,:,0])>=AMPLITUDE_SATURATION)
                        ## if manipulating img directly (as is done on the basis of range min/max below)
                        # saturation indices must be identified before zero distance indices
                        img,outside_range_indices = phase_to_distance(img,self.ambiguity_distance, range_max = self.range_max,
                                                                    range_min = self.range_min)

                        zero_distance_indices = np.where(np.squeeze(img)==0)
                        img = convert_matrix_image(img,cmap= 'jet_r',  clim_min=self.range_min, clim_max=self.range_max,
                                                   saturation_indices = amplitude_saturated_indices,
                                                   no_data_indices = zero_distance_indices,
                                                   outside_range_indices = outside_range_indices, colorize = self.colorize)
                        cnt = self.plot(img,cnt,camera_index,'jet_r')

                        img = np.squeeze(frame)[:,:,1]
                        amplitude_saturated_indices = np.where(np.squeeze(frame[:,:,1])>=AMPLITUDE_SATURATION)
                        img, outside_range_indices = phase_to_distance(img,self.ambiguity_distance, range_max = self.range_max,
                                                                    range_min = self.range_min)
                        zero_distance_indices = np.where(np.squeeze(img)==0)
                        img = convert_matrix_image(img,cmap= 'jet_r',  clim_min=self.range_min, clim_max=self.range_max,
                                                   saturation_indices = amplitude_saturated_indices,
                                                   no_data_indices = zero_distance_indices,
                                                   outside_range_indices = outside_range_indices,colorize = self.colorize)
                        cnt = self.plot(img,cnt,camera_index,'jet_r')

                        img = np.squeeze(frame)[:,:,2]
                        amplitude_saturated_indices = np.where(np.squeeze(frame[:,:,2])>=AMPLITUDE_SATURATION)
                        img, outside_range_indices = phase_to_distance(img,self.ambiguity_distance, range_max = self.range_max,
                                                                    range_min = self.range_min)
                        zero_distance_indices = np.where(np.squeeze(img)==0)
                        img = convert_matrix_image(img,cmap= 'jet_r',  clim_min=self.range_min, clim_max=self.range_max,
                                                   saturation_indices = amplitude_saturated_indices,
                                                   no_data_indices = zero_distance_indices,
                                                   outside_range_indices = outside_range_indices,colorize = self.colorize)
                        cnt = self.plot(img,cnt,camera_index,'jet_r')

                    raw_phase = np.squeeze(frame)[:,:,3]
                    raw_ampl = np.squeeze(frame)[:,:,4]
                    # this uses amplitude to deinterlace distance (img)
                    raw_ampl, raw_phase = getComparedHDRAmplitudeImage_vectorized(raw_ampl, raw_phase)
                    rgb_img = None
                    if self.rgb and data[4] is not None:
                        rgb_img = data[4]
                    rgb_img,raw_phase,raw_ampl = camera_calibrations(rgb=rgb_img,depth=raw_phase,ampl=raw_ampl,camera_version = self.camera_version)
                    
                    #amplitude_saturated_indices = np.where(np.squeeze(raw_ampl>=self.AMPLITUDE_SATURATION))
                    #raw_phase = img.copy()
                    #amplitude_low_indices = np.where(raw_ampl == self.LOW_AMPLITUDE)
                    #amplitude_beyond_range_indices = np.where(np.logical_or(raw_ampl < self.ampl_min,
                    #                                                        np.logical_and(raw_ampl>self.MAX_AMPLITUDE,raw_ampl<self.LOW_AMPLITUDE)))
                    
                    #phase_min = self.range_min*self.MAX_PHASE/self.ambiguity_distance
                    #phase_max = self.range_max*self.MAX_PHASE/self.ambiguity_distance
                    #phase_beyond_range_indices = np.where(np.logical_and(np.logical_or(raw_phase<phase_min, raw_phase>phase_max),raw_phase>0))
                    filtered_phase, thresholded_ampl, indices = threshold_filter(raw_phase = raw_phase, raw_ampl = raw_ampl, reflectivity_thresh= self.reflectivity_thresh,
                                                                             range_max = self.range_max, range_min = self.range_min, ampl_min = self.ampl_min,
                                                                             filter_params = self.filter_params, ambiguity_distance = self.ambiguity_distance)
         
                    
                    if self.view_as is not None:
                        dist, _ = phase_to_distance(filtered_phase,self.ambiguity_distance, range_max = self.range_max,
                                                                    range_min = self.range_min)
                        #zero_distance_indices = np.where(np.squeeze(img)==0)
                        outside_range_indices = indices['amplitude_beyond_range']
                        for k in indices.keys():
                            if 'beyond_range' in k and k != 'amplitude_beyond_range':
                                outside_range_indices = (np.concatenate((outside_range_indices[0], indices[k][0])),
                                                         np.concatenate((outside_range_indices[1], indices[k][1])))
                        img = convert_matrix_image(dist,cmap= 'jet_r',  clim_min=self.range_min, clim_max=self.range_max,
                                                   saturation_indices = indices['amplitude_saturated'],
                                                   no_data_indices = indices['amplitude_low'],
                                                   outside_range_indices = outside_range_indices,colorize = self.colorize)
                        
                        cnt = self.plot(img,cnt,camera_index,'jet_r')
                        
                        if self.rgb and rgb_img is not None:
                            # this was needed here because in plot all data gets rotated I am going to get all the flip etc here
                            #_rgb_img = np.transpose(rgb_img,(1,0,2))
                            # if self.camera_version == 'oyla_1_camera_v0':
                            #     #pure hack also because saving may not have been correct
                            #     _rgb_img = np.flipud(np.fliplr(_rgb_img))
                            cnt = self.plot(_rgb_img,cnt,camera_index)

                elif imaging_type[ind] == 'DCS_2':
                    if self.view_as is not None:
                        img = np.squeeze(frame)[:,:,0]
                        cnt = self.plot(img,cnt,camera_index,'viridis')
                        img = np.squeeze(frame)[:,:,1]
                        cnt = self.plot(img,cnt,camera_index,'viridis')
                    
            
            for i in range(self.number_images):
                # here camera is attached to a view and its attahced only once when first frame is available
                if self.updated_canvas[i] and self.got_image[i]:
                    self.vbs[i].camera = scene.PanZoomCamera(aspect=1)
                    #self.vbs[i].camera.zoom = 2.0
                    self.vbs[i].camera.reset()
                    #self.vbs[i].camera.set_range()
                    # to make sure that view gets resized with input (binning or not)
                    if i == self.number_images-1 and self.rgb:
                        self.vbs[i].camera.set_range(x=(0,rgb_img.shape[1]),y=(0,rgb_img.shape[0]))
                    else:
                        self.vbs[i].camera.set_range(x=(0,fp.shape[1]),y=(0,fp.shape[0]))
                    self.updated_canvas[i] = False
                
            
                
            self.update()#pass
        return filtered_phase, thresholded_ampl,rgb_img
            
    def plot(self, img, cnt, camera_index,cmap='viridis',clim=(0.1,9.0)):
        '''
        Actual plotting function, 
        '''
        # With stitching camer_index ==0, what is need for number_chips? that is it may be needed only in case of multiple
        # cameras with no stitching
        #import matplotlib.pyplot as plt
        #plt.imsave('test.png',img)

        
        if self.camera_version =='oyla_2_camera':
            
            if camera_index ==1:
                self.images[cnt+self.number_chips*camera_index].set_data(np.fliplr((img)))
            else:
                self.images[cnt+self.number_chips*camera_index].set_data((np.flipud(img)))
        elif 'oyla_1_camera' in self.camera_version:
            # to plot correctly the image has to be first inverted up
            self.images[cnt+self.number_chips*camera_index].set_data(np.flipud(img))
        elif self.camera_version == 'espros':
            self.images[cnt+self.number_chips*camera_index].set_data(np.flipud(img))

        self.images[cnt+self.number_chips*camera_index].cmap = cmap
        #print('xxxx',self.number_images,cnt+self.number_chips*camera_index,camera_index,self.number_chips,cnt)
        self.got_image[cnt+self.number_chips*camera_index] = True
        self.colorbars[cnt+self.number_chips*camera_index].cmap = self.images[cnt+self.number_chips*camera_index].cmap
        #print(self.colorbars[cnt+self.number_chips*camera_index].cmap.colors,cmap)
        self.colorbars[cnt+self.number_chips*camera_index].clim  = clim
        return cnt+1
