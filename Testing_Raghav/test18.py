import cv2
import matplotlib.pyplot as plt
def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    #return str(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)),str(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    return 0

cam = cv2.VideoCapture(0)
#cam.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

for x in range(120,1081,120):
    for y in range(160,1921,160):
        set_res(cam,x,y)
        r, frame = cam.read()
        if r:
            print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            plt.imsave('./'+str(frame.shape[0])+'_'+str(frame.shape[1])+'.png',frame)
