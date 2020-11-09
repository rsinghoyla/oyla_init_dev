##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Creation:          20.07.2019                                                                  #
# Version:           1.0                                                                         #
# Version:           2.0 20.10.2019
# Revision history:  Initial version, revised multipl times                                      #
# Description                                                                                    #
#        main  script for  getting data and visualization;
#        run multiple epochs                                                                     #
#        store data ONLY for last epoch and only for DCS, HDR, Dist_Ampl
#        pcl and range viz is a cmd line flag, pcl works only on dist_ampl or HDR
##################################################################################################
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
import time
import numpy as np
from queue import Queue
from queue import Empty
import sys
import shutil
import copy
import cv2
import argparse
import datetime
import os
import threading

# this is for Windows Users though would be better to do export PYTHONPATH
sys.path.append("C:/Users/Oyla1/Documents/GitHub/our_python_dev/Testing_Raghav/")
sys.path.append("C:/Users/Oyla1/Documents/GitHub/our_python_dev/")


from oyla.mvc.utils import  stitch_frame, write_files, CAMERA_VERSION
from oyla.utils import read_csv_parameters, some_common_utility,  AMBIGUITY_DISTANCE_LUT, MODULATION_FREQUENCIES
from oyla.mvc.model import Camera
#import oyla.mvc.pcl.views as pcl
#import oyla.mvc.range.views as rv
from oyla.mvc.views import MainWindow

# A lot of viz stuff wa Inspired from this  thread
#https://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt
class SaveThread(threading.Thread):
    ############################################################
    # https://stackoverflow.com/questions/25904537/how-do-i-send-data-to-a-running-python-thread
    # save data to file using queue to pass stored data to thread; deletes data after writing
    ############################################################
    def __init__(self, queue, args=(), kwargs=None):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.queue = queue
        self.daemon = True
        #self.receive_messages = args[0]

    def run(self):
        while True:
            try:
                val = self.queue.get()
                save_path = val[0]
                key = val[1]
                stored_data = val[2]
                frame_number = val[3]
                write_files(stored_data,key,save_path,fno=frame_number)
                del stored_data
            except Empty:
                break
            #self.queue.task_done()
                                

class MovieThread(QtCore.QThread):
    def __init__(self, cameras,queue, imaging_type,number_chips=1, rgbcamera=None,name = None):
    #####################################################################################
    # initializes the thread for getting frames of data (like a movie)
    # input: cameras, queue , imaging_type, number of chips, rgbCamera available or not
    # output: run will pop the next element in Queue, and it will get the frame for the corresponding camera and imaging_type
    #####################################################################################
        super().__init__()
        self.number_frames = 0
        self.cameras = cameras
        self.queue = queue
        self.number_chips = number_chips
        self.imaging_type = imaging_type
        self.rgbcamera = rgbcamera
        self.name = name


    dataChanged = QtCore.pyqtSignal(list,list,int)
    def run(self):
        #####################################################################################
        # running till queue is empty. we are not blocking on the queue -- I could not get that and
        # the plotting thread to work together
        # 
        #####################################################################################
        then = time.time()
        
        
        while True and self.queue is not None:
            
            try:
                # Get the camera and frame_number from the queue 
                chip, frame_number = self.queue.get(False)
                print("On thread %s sending frame_number %d to chip %s"%(self.name,frame_number,chip))
            except Empty:
                now = time.time()
                print("Frame rate: ", (self.number_frames/(now-then)))
                self.number_frames = 0
                break
            try:                
                # Logging pre command, and post command time along with camera index, and data
                # camera here is the "chip"
                # SINGLE CAMERA to DUAL SETUP TEST:
                #time.sleep(1*np.random.rand())
                
                
                # take rgb frame before depth camera -- discussion with Sid, also in legacy data this may be 640,480 zeros
                rgbframe = np.zeros((0,0,3))
                if self.rgbcamera is not None:
                    ret, rgbframe = self.rgbcamera.read()
                    if ret:
                        rgbframe = cv2.cvtColor(rgbframe, cv2.COLOR_BGR2RGB)
                    else:
                        print('NO RGB',rgbframe)

                pre_time = time.time()
                data = []
                #imgaing_type could be ['Gray','Dist_Ampl']
                for i,it in enumerate(self.imaging_type):
                    # I am sending frame_number to model, onward to epc_lib such that data from camera is tagged with the frame number when it returns
                    _data = self.cameras[chip].get_frame(it,frame_number)
                    data.append(_data)    
                post_time = time.time()
                print('On thread %s got frame_number %d from depth server '%(self.name,_data[2]))
                
                # so output is time when data is got, when request for dat is sent, data, chip being used, adn rgbframe
                _res = [post_time, pre_time, data, chip, rgbframe, self.name]
                # _res is data ++ imaging_type, number_chips which are parameters which are being passed to update_data to store;
                # this is because update_data is not part of Controller class; see belpw
                # here the signal is emited and view is updated
                self.dataChanged.emit(_res,self.imaging_type,self.number_chips)
                #not sure what this sleep helps in -- maybe window, mouse control
                QtCore.QThread.msleep(10) 
            
            finally:
                # this is needed to signal done
                self.queue.task_done()
            self.number_frames += 1
            
        now = time.time()


class Controller():
    def __init__(self,cameras,parameters,window,view_as = 'range', rgb=False, filter_parameters = {}):
        #####################################################################################
        # the main controller class that controls the data thread, and its interface with window (view)
        # for multiple epochs there is a restat function that at begining of epoch initializes the cameras
        # adds the cameras to the queue in order that they are to be called
        # and then starts as many new threads as needed
        # I did check that python garabage collected threads
        #####################################################################################
        self.cameras = cameras
        self.queue = Queue()
        self.window = window
        self.parameters = parameters
        self.params = self.parameters['param']
        self.number_epochs = int(self.params['number_epochs'][0])
        assert 'camera_version' in self.params, "no camera version in parameters file"
        self.camera_version = self.params['camera_version'][0]
        assert self.camera_version in CAMERA_VERSION, "Not supported camera version"
        self.epoch_number = 0
        self.view_as = view_as
        self.rgb = rgb
        if self.rgb:
            self.rgbcamera = cv2.VideoCapture(0)
            self.rgbcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.rgbcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        else:
            self.rgbcamera = None
        self.filter_parameters = filter_parameters
        
    def start_movie(self,imaging_type,number_chips=1,name = None):
        # starts a new movie thread and connects the data changed to updation
        movie_thread = MovieThread(self.cameras,self.queue,imaging_type,number_chips=number_chips, rgbcamera=self.rgbcamera, name = name)
        movie_thread.start()
        movie_thread.dataChanged.connect(update_data)
        return movie_thread    
        
    def restart(self):
        #####################################################################################
        # The main function -- everything happens here
        #####################################################################################
        print('Restarting')
        global prev_frames
        global stored_data
        prev_frames = {}
        stored_data = {}
        # get the server and image_class for this epoch; TODO fix if there are multiple chips per epoch
        chips = self.params['chip_id'][self.epoch_number].split(',')
        for chip in chips:
            camera = self.cameras[chip]
            # execute adaptive commands for this epoch on this server
            camera.server_commands(self.parameters['adaptive_cmd'], self.epoch_number)
            # execute commands that are dependent on params;
            #ISSUE: enable compensation and correctDRNU -- seems like enableCompensation would override
            camera.logical_server_commands(self.params, self.epoch_number)
                        
        # get imagining type; if two or more imaging type are aquired per frame they should be seperated
        # by a +; a split on + gets all the imaging_type for to be acquired for this  epoch
        # also note that if imaging_type =X+y then X type will be imaged first and then y type.
        imaging_type = self.params['imaging_type'][self.epoch_number].split('+')

        #get imaging_mode; have to fix this after I understand MGX better
        imaging_mode = self.params['imaging_mode'][self.epoch_number]

        #to take care of enablePiDelay==0 in which case only 2 DCS are collected
        if 'DCS' in imaging_type and self.parameters['adaptive_cmd']['enablePiDelay'][self.epoch_number] == '0':
                    imaging_type[imaging_type.index('DCS')] = 'DCS_2'

        if imaging_mode == 'HDR':
            imaging_type = ['HDR'] #a hack for now; but also established

        #######################################################################
        # I wanted to remove this -- but its unclear where it goes; these parameters actually sit between model and view
        # they could be more view specific but then view becomes dependent on parameters, epoch_number etc. which is not clean
        # sent it to utils so that controller and controller_read_dist are doing the same thing
        #######################################################################
        ambiguity_distance, range_max, range_min, saturation_flag, adc_flag, mod_freq,ampl_min,reflectivity_thresh = some_common_utility(self.parameters,self.epoch_number)

       
        # CHANGE -- THIS WAS READING FILTER PARAMETERS from paramaters_csv and  Now we are reading from filter_csv
        filter_params = {}
        if 'filter_cmd' in filter_parameters:
            for k in filter_parameters['filter_cmd'].keys():
                try:
                    filter_params[k] = filter_parameters['filter_cmd'][k][self.epoch_number]
                except ValueError:
                    pass
                print(filter_params)

        
        #by default each imaging_type in X+y will be plotted in a column
        number_images = len(imaging_type)

        #for each imaging_type depending on number of frames we will adjust the rows,columns of subplots
        if 'Dist_Ampl' in imaging_type:
            number_images += 1
        if 'DCS' in imaging_type:
            number_images += 3
        if 'DCS_2' in imaging_type:
            number_images += 1
        if 'HDR' in imaging_type:
            number_images += 3
        if self.rgb:
            number_images += 1
        # Key change to incorporate multiple cameras/chips, if NOT stiching
        #number_images *= len(chips)
        
        print("Imaging type and Number of Images",imaging_type,number_images)
        # assertion to ensure that only Dist_Ampl data
        # This is the function that sets up the number of plots for the next epoch, also plotting specific
        # parameters are passed here.
        # if view_as is None the window and canvas for range will be created but no plotting will be done; so no viz
        if self.view_as == 'pcl':
            assert imaging_type[0] == 'Dist_Ampl' or imaging_type[0] == 'HDR',"pcl only Dist_Ampl, HDR"
            # this is different from generic case in range -- here only one "image" in canvas
            self.window.canvas.update_canvas(number_chips = len(chips),
                                             ambiguity_distance = ambiguity_distance,
                                             range_max = range_max, range_min = range_min, 
                                             saturation_flag = saturation_flag,adc_flag = adc_flag, mod_freq = mod_freq,
                                             filter_params = filter_params,rgb = self.rgb,ampl_min = ampl_min,
                                             reflectivity_thresh = reflectivity_thresh,camera_version = self.camera_version)
        else:
            self.window.canvas.update_canvas(number_images =number_images, number_chips = len(chips),
                                             ambiguity_distance = ambiguity_distance,
                                             range_max = range_max, range_min = range_min, 
                                             saturation_flag = saturation_flag,adc_flag = adc_flag,mod_freq = mod_freq,
                                             filter_params = filter_params, rgb = self.rgb,view_as = self.view_as,ampl_min = ampl_min,
                                             reflectivity_thresh = reflectivity_thresh,camera_version = self.camera_version)
            
        # add the chips to the queue for the number of frames in this epoch
        # any order can be used to put in chip but has to match the laser firing order
        for iterator in range(int(self.params['number_frames_epoch'][self.epoch_number])):
            for chip in chips:
                self.queue.put([chip,iterator])

                
        ##############################################################################################
        # Here multiple threads, as many as the number of cameras, will be started
        # this should make frame rate much higher.
        # TESTED -- ABOVE IS SINGLE THREAD for camera and below is multiple thread -- ABOVE look at older code before 153 commit
        self.movie_thread = {}
        for chip in chips:
            self.movie_thread[chip] = self.start_movie(imaging_type,len(chips),name = str(chip)+'T')
            # this is mainly to cater to multiple epochs -- may not be needded if only one epoch
            if self.epoch_number == self.number_epochs-1:
                self.movie_thread[chip].finished.connect(self.finished)
            else:
                 self.movie_thread[chip].finished.connect(self.restart)
        
        ##############################################################################################
        self.epoch_number += 1

    def finished(self):
        if self.rgbcamera is not None:
            time.sleep(1)
            self.rgbcamera.release()
            print("released rgb")
        print('FINISHED')

# If i make this part of controller I get a null message on connect -- have to debug ... the down side is teh global variables and parameters passed in signal
@QtCore.pyqtSlot(list,list,int)
def update_data(data,imaging_type,number_chips):
    ###################################################################
    # Signal is sent here and in case of multiple cameras stitching happens here and then data sent to visualization thread
    # Things that could slow down frame rate -- deepcopy (twice), stitching and possibly filtering.. as of now all inbuilt functions so shoudl be optkmized
    #####################################################################
    global prev_frames
    global stored_data
    global view_as
    global online_save
    global save_path
    global filtered_save_path
    global save_queue
    global save_thread
    global filters_on
    global debug
    
    frame_number  = data[2][0][2] #there has to be atleast one imaging type therefore index 0
    camera_chip = data[3]
    # This is needed for stitching, else camera (indiviual) data gets overwritten by stitching
    data_c = copy.deepcopy(data)

    # thought about moving filtering here, so that filtering can be done without visualization,
    # but its not that elegant in the sense we have to have "if and for" conditions for different types of images
    # repeat of what is done in view. So I am going to let filtering and visualize work together; I stop plotyting in case of no viz and let canvas come up

    # Initialize stored data
    if camera_chip not in stored_data:
        stored_data[camera_chip] = {}
    # important that stitching is on for multiple number chips
    if number_chips > 1 and 'stitched' not in stored_data:
        stored_data['stitched'] = {}
    if filters_on and 'filtered' not in stored_data:
        stored_data['filtered'] = {}
        
    if frame_number not in stored_data[camera_chip]:
        stored_data[camera_chip][frame_number] = data
    else:
        print("error",number_chips)

    
    # A way of keeping track that how many of the chips have been recieved for this frame;  (deleted on stitching)
    if frame_number not in prev_frames:
        prev_frames[frame_number] = 1.0/number_chips
    else:
        prev_frames[frame_number] += 1.0/number_chips

    # got_frame_for_all_chips
    if prev_frames[frame_number] == 1.0:
        if number_chips == 2:
            #currently stiching only for two cameras
            for ind in range(len(imaging_type)):
                frame1 = stored_data['1'][frame_number][2][ind][0]
                frame2 = stored_data['2'][frame_number][2][ind][0]
                # Stitch and store in data_c
                frame = stitch_frame(frame1,frame2)
                data_c[2][ind][0] = frame
        
            #This is being done because after stitching its assumed there is only chip '1' for display purposes
            data_c[3] = '1'
            # here is where the stitched data gets updated
            stored_data['stitched'][frame_number] = data_c
            stored_data['stitched'][frame_number].append(time.time())
            
        if number_chips == 1:
            # Stored data updated with time in case of single camera; just to be consistent
            stored_data[camera_chip][frame_number].append(time.time())    

        # update view using data_c which is either stitched or single camera; filtered_phase is non None if filtering on and returned from view
        #filtered_phase = None
        #filtered_rgb = None
        filtered_phase, filtered_ampl,filtered_rgb = window.update_data(data_c,imaging_type)

        #update in case of filter
        if filters_on:
            data_f = copy.deepcopy(data_c)
            #d = data_f[2][0][0]
            tmp = np.stack((filtered_phase,filtered_ampl),axis=2)
            _tmp = []
            for i,_a in enumerate(data_f[2][0]):
                if i == 0:
                    _a = tmp
                _tmp.append(_a)
            data_f[2][0] = np.asarray(_tmp)
            if filtered_rgb is not None:
                data_f[4] = np.resize(data_f[4],filtered_rgb.shape)
                data_f[4] = filtered_rgb
            #d[:,:,0] = filtered_phase
            #d[:,:,1] = filtered_ampl
            stored_data['filtered'][frame_number] = data_f
            stored_data['filtered'][frame_number].append(time.time())
            
        #this is for online saving, this will put data in the save queue for save thread to work
        # assumption in save thread is that only one frame is available in stored_data, all others are deleted.
        # note that save thread will delete data after it has written to a file
        if online_save:
            if number_chips == 1:
                val = [save_path, '1', stored_data['1'][frame_number], frame_number]
            else:
                val = [save_path, 'stitched', stored_data['stitched'][frame_number], frame_number]
            save_queue.put(val)
            if number_chips == 1:
                del stored_data['1'][frame_number]
            else:
                del stored_data['stitched'][frame_number]
            if filters_on:
               val = [filtered_save_path, 'filtered', stored_data['filtered'][frame_number], frame_number]
               save_queue.put(val)
               del stored_data['filtered'][frame_number]
               
        #clean up unnecessary frames, for two cameras its frame 1,2 and if saving then 'stitched' also note online saving will not delete these as they are not stored
        if number_chips > 1 and not debug:
            del stored_data['1'][frame_number]
            del stored_data['2'][frame_number]
                
        del prev_frames[frame_number]


if __name__ == "__main__":


    # setup the chips dictionary which is key: chip_id, value: chip_ip
    # setup csv_file path
    
    chip_IPs = {}
    #chip_IPs['1'] = "10.10.31.184"
    #chip_IPs['2'] = "10.10.31.186"
    chip_IPs['1'] = "192.168.7.2"
    chip_IPs['2'] = "192.168.7.2" 
    csv_file = "../../../DME_Python_3_Current/oyla/FinalParams_v4-Table 2 R.csv"

    # For ambiguity distance LUT; currently ESPROS has following frequencies

    # script setup only upto here
    ##############################################################################
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--view_as",type = str, default = "range")
    parser.add_argument("--save_path",type = str, default = "")
    parser.add_argument("--rgb", type = bool, default = False)
    parser.add_argument("--parameters_csv",type=str,required=True)
    parser.add_argument("--filters_csv",type=str,default=None)
    parser.add_argument("--online_save",type=bool,default=False)
    parser.add_argument("--debug",type=bool,default=False)
    #parser.add_argument("--camera_version",type = str, default = None)
    
    args = parser.parse_args()
    csv_file = args.parameters_csv
    filters_on = args.filters_csv is not None
    view_as = args.view_as
    online_save = args.online_save
    debug = args.debug
    save = len(args.save_path)>0
    prev_frames = {}
    stored_data = {}
    
    # read parameters from csv file
    parameters = read_csv_parameters(csv_file)
  

    filter_parameters = {}
    if args.filters_csv is not None:
        filter_parameters = read_csv_parameters(args.filters_csv)
        assert filter_parameters['param']['number_epochs'][0] == parameters['param']['number_epochs'][0]

    save_queue = None
    save_thread = None
    if online_save:
        save_queue = Queue()
        save_thread = SaveThread(queue = save_queue)
        save_thread.start()
    
    # set the camera class for each chip
    cameras = {}
    for c in chip_IPs:
        cameras[c] = Camera(chip_IPs[c])    

    # do default command setup on each chip, once.
    for c in chip_IPs:
        cameras[c].server_commands(parameters['default_cmd'],0)

    # this has to be always in this order, QApp and then window which is a QWidget
    app = QApplication([])
    # Sending chip IPS to window because epc_lib will return server IP; not chip_IP dictionary as above, or chip_id as in csv
    # change from range in that fixed number of views always==1; also colorbar set
    # change on Dec 6 -- a common window will all the parameters in setup widget
    window = MainWindow(args.view_as, rgb = args.rgb, show_colorbar=[0,1],save = save,live=True)
            
    window.resize(1300,700)
    # rgb here so that thread collects rgb
    controller = Controller(cameras, parameters, window, args.view_as, rgb=args.rgb, filter_parameters = filter_parameters)
    
    if save:
        assert int(parameters['param']['number_epochs'][0]) == 1, "ONLY SUPPORTS ONE EPOCH SAVING"
        #have to fix this, TBD working for multi epochs also but only last epoch is stored.
        imaging_type = parameters['param']['imaging_type'][0].split('+')
        assert imaging_type[0] == 'Dist_Ampl' or 'HDR' or 'DCS', "Dist_Ampl, HDR, DCS only"
        
        save_path = args.save_path
        t = datetime.datetime.now()    
        save_path = save_path+'_data_{:%B_%d_%H_%M_%S}'.format(t)
        if not os.path.isdir(save_path): os.mkdir(save_path)
        shutil.copy(csv_file,save_path+'/parameters.csv')
        
        if args.filters_csv is not None:
            filtered_save_path = save_path+'/filtered_data_{:%B_%d_%H_%M_%S}'.format(t)
            if not os.path.isdir(filtered_save_path): os.mkdir(filtered_save_path)
            shutil.copy(csv_file,filtered_save_path+'/parameters.csv')
            shutil.copy(args.filters_csv,filtered_save_path+'/filter_params.csv')
            
    
    controller.restart()
    window.show()
    
    app.exit(app.exec_())
  
    # finding mean difference in sought frame, got frame, stitching frame, time
    for key in stored_data.keys():
        if key == 'filtered':
            continue
        diff = 0
        diff1 = 0
        diff2 = 0
        for fno in stored_data[key].keys():
            if fno > 0:
                try:
                    diff1 += (stored_data[key][fno][0]-stored_data[key][fno-1][0])
                    diff += (stored_data[key][fno][1]-stored_data[key][fno-1][1])
                    if len(stored_data[key][fno])>7:
                        # there is stitching time info available; could be issue when debug is on
                        diff2 += (stored_data[key][fno][6]-stored_data[key][fno-1][6])
                except KeyError:
                    # this will happen in case of closing window when online save is on 
                    print(key,fno)
                    print(stored_data.keys())

        if len(list(stored_data[key].keys()))>0:
            # condition needed to avoid key that may not have data after deletion
            print("Sought Frame diff ms for chip ",key, diff1/len(list(stored_data[key].keys())))
            print("Got Frame diff ms for chip ",key, diff/len(list(stored_data[key].keys())))
            print("Stitch Frame diff ms for chip ",key, diff2/len(list(stored_data[key].keys())))
        
    #for key in stored_data.keys():
    #    print(stored_data.keys())

    	# else:
        #    print("deleted ",key)
    if save and not online_save:
        
        for key in stored_data.keys():            
            # a bit convoluted clean up beccause keys '1', '2' and 'filtered' may always be there
            if key == 'filtered':
                write_files(stored_data[key],key,filtered_save_path)
            else:
                if 'stitched' in stored_data and key != 'stitched' and not debug:
                    #debug everything has to be written
                    continue
                write_files(stored_data[key],key,save_path)
