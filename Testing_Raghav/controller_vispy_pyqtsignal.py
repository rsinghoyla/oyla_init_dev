from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
from views_vispy_pyqtsignal import MainWindow
from model_espros_660 import Camera 
import time
import numpy as np
from queue import Queue
from queue import Empty
from utils_black import read_csv_parameters
#https://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt

class MovieThread(QtCore.QThread):
    def __init__(self, queue, imaging_type):
        super().__init__()
        #self.camera = camera
        self.num_frames = 0
        self.queue = queue
        self.imaging_type = imaging_type
        
        self.results = {}
        #print(self)
    dataChanged = QtCore.pyqtSignal(list,list)
    def run(self):
        then = time.time()
        # for _ in range(self.num_frames):
        #     print('send server command at ',time.time())
        #     data = self.camera.get_frame()#sensor_data(4, 10, 10)
        #     self.dataChanged.emit(data)
        #     QtCore.QThread.msleep(1)
        
        while True and self.queue is not None:
            # Get the camera from the queue 
            try:
                camera = self.queue.get(False)
            except Empty:
                now = time.time()
                print("Frame rate: ", (self.num_frames/(now-then)))
                self.num_frames = 0
                break
            try:
                if camera not in self.results:
                    self.results[camera] = []
                #print(image)
                #Logging pre command, and post command time in result
                pre_time = time.time()
                print('send server command at ',time.time())
                data = []
                for i,it in enumerate(self.imaging_type):
                    _im = camera.get_frame(it)
    
                    data.append(_im)#sensor_data(4, 10, 10)
                #print(len(data))
                #data = np.asarray(data)
                post_time = time.time()
                J = [post_time, pre_time, data, list(self.results.keys()).index(camera)]
                
                self.results[camera].append(J)
                self.dataChanged.emit(J,self.imaging_type)                
                QtCore.QThread.msleep(1)
            
            finally:
                self.queue.task_done()
            self.num_frames += 1
            
        now = time.time()
        #print("Frame rate: ", (self.num_frames/(now-then)))

class Controller():
    def __init__(self,cameras,parameters,window):
        self.cameras = cameras
        #self.camera.initialize()
        self.queue = Queue()
        self.window = window
        self.parameters = parameters
        self.params = self.parameters['param']
        self.num_epochs = int(self.params['number_epochs'][0])
        self.epoch_number = 0
        
    def start_movie(self,imaging_type):
        movie_thread = MovieThread(self.queue,imaging_type)
        movie_thread.start()
        movie_thread.dataChanged.connect(self.window.update_data)
        return movie_thread    
        
    def restart(self):
        print('Restarting')
        # get the server and image_class for this epoch; TODO fix if there are multiple chips per epoch
        chips = self.params['chip_id'][self.epoch_number].split(',')
        for chip in chips:
            camera = self.cameras[chip]
            # execute adaptive commands for this epoch on this server
            camera.server_commands(self.parameters['adaptive_cmd'], self.epoch_number)
            # execute commands that are dependent on params;
            camera.logical_server_commands(self.params, self.epoch_number)

        # get imagining type; if two or more imaging type are aquired per frame they should be seperated
        # by a +; a split on + gets all the imaging_type for to be acquired for this  epoch
        # also note that if imaging_type =X+y then X type will be imaged first and then y type.
        imaging_type = params['imaging_type'][self.epoch_number].split('+')

        #get imaging_mode; have to fix this after I understand MGX better
        imaging_mode = params['imaging_mode'][self.epoch_number]

        #to take care of enablePiDelay==0 in which case only 2 DCS are collected
        if 'DCS' in imaging_type and parameters['adaptive_cmd']['enablePiDelay'][self.epoch_number] == '0':
                    imaging_type[imaging_type.index('DCS')] = 'DCS_2'

        if imaging_mode == 'HDR':
            imaging_type = ['HDR'] #a hack for now

        #######################################################################
        try:
            ambiguity_distance = ambiguity_distance_LUT[int(parameters['adaptive_cmd']
                                                            ['setModulationFrequency'][self.epoch_number])]
        except ValueError:
            ambiguity_distance = None
        try:
            range_max = float(parameters['adaptive_cmd']['range_max'][self.epoch_number])
        except ValueError:
            range_max = None
        try:
            range_min = float(parameters['adaptive_cmd']['range_min'][self.epoch_number])
        except ValueError:
            range_min = None
        try:
            amplitude_min = float(parameters['adaptive_cmd']['setMinAmplitude'][self.epoch_number])
        except ValueError:
            amplitude_min = None
        try:
            saturation_flag = int(parameters['adaptive_cmd']['enableSaturation'][self.epoch_number])
        except ValueError:
            saturation_flag = None
        try:
            adc_flag = int(parameters['adaptive_cmd']['enableAdcOverflow'][self.epoch_number])
        except ValueError:
            adc_flag = None
            
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
        number_images *= len(chips)
        print(imaging_type,number_images)
        
        ##############################################################################################
        
        self.window.canvas.update_canvas(number_images =number_images, number_cameras = len(chips),
                                         ambiguity_distance = ambiguity_distance,
                                         range_max = range_max, range_min = range_min, 
                                         saturation_flag = saturation_flag,adc_flag = adc_flag)

        for iterator in range(int(params['number_frames_epoch'][self.epoch_number])):
            for chip in chips:
                self.queue.put(self.cameras[chip])

        self.movie_thread = {}
        for chip in chips:
            self.movie_thread[chip] = self.start_movie(imaging_type)

            if self.epoch_number == self.num_epochs-1:
                self.movie_thread[chip].finished.connect(self.finished)
            else:
                self.movie_thread[chip].finished.connect(self.restart)
        self.epoch_number += 1

    def finished(self):
        print('FINISHES')
    

    

if __name__ == "__main__":


    # setup the chips dictionary which is key: chip_id, value: chip_ip
    # setup csv_file path
    
    chips = {}
    chips['1'] = "192.168.7.2"
    chips['2'] = "192.168.7.2" 
    csv_file = "../DME_Python_3_Current/oyla/FinalParams_v4-Table 2.csv"

    # For ambiguity distance LUT; currently ESPROS has following frequencies (server.sendCommand("getModulationFrequencies")[::2])
    modulation_frequencies = np.asarray([24000, 12000, 6000, 3000, 1500, 750, 24000])*1000 # KHz to Hz
    #
    speed_light = 150000000.0
    ambiguity_distance_LUT = speed_light/modulation_frequencies #in meters
    #setup only upto here
    ##############################################################################

    parameters = read_csv_parameters(csv_file)

    # set the camera class for each chip
    cameras = {}
    for c in chips:
        cameras[c] = Camera(chips[c])    

    # do default command setup on each chip, once.
    for c in chips:
        cameras[c].server_commands(parameters['default_cmd'],0)

    params = parameters['param']

    app = QApplication([])
    window = MainWindow()

    #window.resize(360*1.5,240*1.5)
    window.resize(1000,800)
    window.show()
    controller = Controller(cameras, parameters, window)
    
    
    
    controller.restart()

    app.exit(app.exec_())
