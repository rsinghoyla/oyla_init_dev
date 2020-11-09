import numpy as np

import cv2
import time

class Camera:
    def __init__(self, cam_num):
        self.cam_num = cam_num
        self.cap = None
        self.last_frame = np.zeros((1,1))

    def initialize(self):
        self.cap = cv2.VideoCapture(self.cam_num)

    def get_frame(self):
        ret, self.last_frame = self.cap.read()
        self.last_frame = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        print('got image',np.mean(self.last_frame),time.time())
        return self.last_frame

    # def acquire_movie(self, num_frames):
    #     import time
    #     then = time.time()
    #     movie = []
    #     for _ in range(num_frames):
    #         print('acquiring image for movie ',time.time())
    #         movie.append(self.get_frame())
    #         #print(np.mean(movie[-1]))
    #     now = time.time()
    #     print("Frame rate: ", (num_frames/(now-then)))
    #     return movie

    def set_brightness(self, value):
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, value)

    def get_brightness(self):
        return self.cap.get(cv2.CAP_PROP_BRIGHTNESS)

    def close_camera(self):
        self.cap.release()

    def __str__(self):
        return 'OpenCV Camera {}'.format(self.cam_num)


if __name__ == '__main__':
    cam = Camera(0)
    cam.initialize()
    print(cam)
    frame = cam.get_frame()
    print(frame)
    cam.set_brightness(1)
    print(cam.get_brightness())
    cam.set_brightness(0.5)
    print(cam.get_brightness())
    cam.close_camera()
