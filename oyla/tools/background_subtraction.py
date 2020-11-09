from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
from skimage.segmentation import felzenszwalb
import matplotlib.pyplot as plt
import matplotlib.animation as animation

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input_dir', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
parser.add_argument('--output_prefix',type=str,default = 'test')
args = parser.parse_args()

## [create]
#create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
## [create]

## [capture]
input = args.input_dir+'/oyla_%04d.png'
capture = cv.VideoCapture(cv.samples.findFileOrKeep(input))
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
## [capture]

#out = cv.VideoWriter('Test.mov',cv.VideoWriter_fourcc('m','p','4','v'), 10, (320,120),1)
fourCC = cv.VideoWriter_fourcc('M','J','P', 'G'); # Important to notice cv2.cv.CV_FOURCC


fig = plt.figure()

number_segments = []
iii = -1
while True:
    iii += 1
    #for iii in range(40):
    ret, frame = capture.read()
    if frame is None:
        break

    ## [apply]
    #update the background model
    fgMask = backSub.apply(frame)
    ## [apply]

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    ## [display_frame_number]

    ## [show]
    #show the current frame and the fg masks
    #print(np.max(frame),np.min(frame),np.unique(fgMask,return_counts=True))
    _fgMask = cv.cvtColor(fgMask,cv.COLOR_GRAY2RGB)
    _frame = np.vstack((frame,_fgMask))
    print(_fgMask.shape)
    fez = felzenszwalb(fgMask,multichannel=False,min_size=40,sigma=0.4,scale = 10)
    masked_frame = 0*frame
    masked_frame[fgMask==255] = frame[fgMask==255]
    print(frame.dtype,np.max(frame))
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    _fez = fez.astype('float32')/np.max(fez)
    _fez = (_fez*255).astype('uint8')
    _fez = cv.cvtColor(_fez,cv.COLOR_GRAY2RGB)
    cv.imshow('Segementation on Mask',_fez)
    _frame = np.vstack((_frame,_fez))
    print("mask",np.unique(fez).shape)
    masked_gray = cv.cvtColor(masked_frame,cv.COLOR_RGB2GRAY)
    fez = felzenszwalb(masked_gray.astype('float32')/np.max(masked_gray+0.000001),multichannel=False,min_size=40,sigma=0.4,scale = 10)
    print("masked",np.unique(fez).shape)
    _fez = fez.astype('float32')/np.max(fez)
    _fez = (_fez*255).astype('uint8')
    _fez = cv.cvtColor(_fez,cv.COLOR_GRAY2RGB)
    cv.imshow('Segementation on Masked',_fez)
    cv.imshow('Masked',masked_gray)
    #print(_frame.shape)
    if iii ==0:
        size = (320,120*3)
        #size = (277,202*3)
        #size = (202,831)
        size = (_frame.shape[1],_frame.shape[0])
        out = cv.VideoWriter("/Users/rsingh/Downloads/"+args.output_prefix+".avi", fourCC, 15.0, size, True)
    out.write(_frame)
    number_segments.append(np.unique(fez).shape[0])
    ## [show]
    
    keyboard = cv.waitKey(300)
    if keyboard == 'q' or keyboard == 27:
        break
    
#ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True,repeat=True)
#plt.show()
#ani.save('/Users/rsingh/Downloads/dynamic_images.mp4')
out.release()
plt.plot(number_segments,'-o')
#plt.show()
plt.savefig("/Users/rsingh/Downloads/"+args.output_prefix+".png")
