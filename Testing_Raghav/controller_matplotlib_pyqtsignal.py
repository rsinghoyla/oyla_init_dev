from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
from views_matplotlib_pyqtsignal import Window
from models_espros_660 import Camera as espros
#from models import Camera as webcam
import time
import numpy as np
from queue import Queue
from queue import Empty
import threading
#https://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt

class MovieThread(QtCore.QThread):
    def __init__(self, camera,num_frames = 0,queue=None):
        super().__init__()
        self.camera = camera
        self.num_frames = num_frames
        self.queue = queue
        if queue is not None:
            self.num_frames = 0
        self.results = {}
        print(self)
        print('tac',threading.active_count())
    dataChanged = QtCore.pyqtSignal(list,list)
    def run(self):
        then = time.time()
        for _ in range(self.num_frames):
            print('send server command at ',time.time())
            data = self.camera.get_frame()#sensor_data(4, 10, 10)
            self.dataChanged.emit(data)
            QtCore.QThread.msleep(1)
        
        while True and self.queue is not None:
            # Get the camera from the queue 
            try:
                camera = self.queue.get(False)
            except Empty:
                break
            try:
                if camera not in self.results:
                    self.results[camera] = []
                #print(image)
                #Logging pre command, and post command time in result
                pre_time = time.time()
                print('send server command at ',time.time())
                data = []
                data.append(camera.get_frame())#sensor_data(4, 10, 10)
                post_time = time.time()
                J = [post_time, pre_time, data]
                self.results[camera].append(J)
                self.dataChanged.emit(data,['HDR'])                
                QtCore.QThread.msleep(1)
            
            finally:
                self.queue.task_done()
            
        now = time.time()
        print("Frame rate: ", (self.num_frames/(now-then)))


def start_movie(camera, num_frames,queue,window):
    movie_thread = MovieThread(camera,num_frames,queue)
    movie_thread.start()
    
    movie_thread.dataChanged.connect(window.update_data)
    return movie_thread

def restart():
    global movie_thread
    for iterator in range(num_frames):
        queue.put(camera)
    print(movie_thread,movie_thread.isFinished())
    movie_thread = start_movie(camera,num_frames=0,queue=queue,window=start_window)
    movie_thread.finished.connect(finished)
    print('tac',threading.active_count())


def finished():
    print(movie_thread,movie_thread.isFinished())
    print('tac',threading.active_count(),threading.current_thread().name)
    print('fin')
    
espros_flag = 1
if espros_flag:
    camera = espros(0)
#else:
#    camera = webcam(0)
camera.initialize()
num_epochs = 1
app = QApplication([])
start_window = Window()

start_window.resize(360*1.5,240*1.5)
start_window.show()
queue = Queue()

for iterator in range(10):
    queue.put(camera)
num_frames= 20
movie_thread = start_movie(camera,num_frames=0,queue=queue,window=start_window)
movie_thread.finished.connect(restart)
import threading
print('tac',threading.active_count())
# for epoch_number in range(num_epochs):
#     print(epoch_number)
    
#     if epoch_number == num_epochs -1:
#         movie_thread.finished.connect(finished)
#     else:
#         movie_thread.finished.connect(restart)
#     #epcoh 0


#queue.join()
app.exit(app.exec_())
