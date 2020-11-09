from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
from views_matplotlib_pyqtsignal import Window
from model_espros_660 import Camera 
import time
import numpy as np
from queue import Queue
from queue import Empty
from utils_black import read_csv_parameters
#https://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt

class MovieThread(QtCore.QThread):
    def __init__(self, camera,queue, imaging_type):
        super().__init__()
        self.camera = camera
        self.num_frames = 0
        self.queue = queue
        self.imaging_type = imaging_type
        
        self.results = {}
        print(self)
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
                    data.append(np.squeeze(camera.get_frame(it)))#sensor_data(4, 10, 10)
                post_time = time.time()
                J = [post_time, pre_time, data]
                self.results[camera].append(J)
                self.dataChanged.emit(data,self.imaging_type)                
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
        movie_thread = MovieThread(self.camera,self.queue,imaging_type)
        movie_thread.start()
        movie_thread.dataChanged.connect(self.window.update_data)
        return movie_thread

    def restart(self):
        print('Restarting')
        # get the server and image_class for this epoch; TODO fix if there are multiple chips per epoch
        chip = self.params['chip_id'][self.epoch_number]
        self.camera = self.cameras[chip]
        # execute adaptive commands for this epoch on this server
        self.camera.server_commands(self.parameters['adaptive_cmd'], self.epoch_number)
        # execute commands that are dependent on params;
        self.camera.logical_server_commands(self.params, self.epoch_number)

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
            
        for iterator in range(int(params['number_frames_epoch'][self.epoch_number])):
            self.queue.put(self.camera)
            
        movie_thread = self.start_movie(imaging_type)

        if self.epoch_number == self.num_epochs-1:
            movie_thread.finished.connect(self.finished)
        else:
            movie_thread.finished.connect(self.restart)
        self.epoch_number += 1

    def finished(self):
        print('FINISHES')
    

    

if __name__ == "__main__":


    # setup the chips dictionary which is key: chip_id, value: chip_ip
    # setup csv_file path
    
    chips = {}
    chips['1'] = "192.168.7.2"
    csv_file = "../DME_Python_3_Current/oyla/FinalParams_v4-Table 1.csv"

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
    window = Window()

    window.resize(360*1.5,240*1.5)
    window.show()
    controller = Controller(cameras, parameters, window)
    
    

    controller.restart()

    app.exit(app.exec_())
