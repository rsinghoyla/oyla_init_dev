##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Creation:          05.08.2020                                                                  #
# Version:           1.0                                                                         #
# Revision history:  Initial version
# Description                                                                                    #
#     This program can be used to make a video of  a set of png files, also can be overlayed
#     with detection boxes which are in yolo format
##################################################################################################
from __future__ import print_function
import cv2 as cv
import os
import pickle as pkl
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='This program can be used to make a video of  a set of png files, also can be overlayed with detection boxes which are in yolo format')
parser.add_argument('--input_dirs', type=str, help='Paths to a sequence of image, multiple directories can be comma seperated ', required=True)
parser.add_argument('--input_file_prefix',type=str,help="images are <image_prefix>_0000.png", default = 'rgb')
parser.add_argument('--output_file_prefix',type=str,default = 'video_')
parser.add_argument("--number_frames",type=int, default = 1000)
parser.add_argument("--yolo_boxes",action="store_true")
args = parser.parse_args()

## [create]
#create Background Subtractor objects

args.output_file_prefix += args.input_file_prefix+'_'
if args.yolo_boxes:
    args.output_file_prefix += 'bbox'
else:
    args.output_file_prefix +='no_bbox'

## [captureyoi
input_dirs = args.input_dirs.split(',')
captures = []
yolo = []
suffix = []
for dirs in input_dirs:
    f = dirs+'/'+args.input_file_prefix+'_%04d.jpg'
    c = cv.VideoCapture(cv.samples.findFileOrKeep(f))
    if not c.isOpened:
        print('Unable to open: ' + dirs)
        exit(0)
    captures.append(c)
    if args.yolo_boxes:
        if os.path.exists(dirs+'/yolo.pkl'):
            with open(dirs+'/yolo.pkl','rb') as fp:
                y = pkl.load(fp)
            su = '/'.join(list(y.keys())[0].split('/')[:-1])
            yolo.append(y)
            suffix.append(su)
if args.yolo_boxes:
    # print(len(captures), len(yolo))
    # print(dirs)
    assert len(captures)==len(yolo)
## [capture]

#out = cv.VideoWriter('Test.mov',cv.VideoWriter_fourcc('m','p','4','v'), 10, (320,120),1)
fourCC = cv.VideoWriter_fourcc('m','p','4','v') #mp4   #('M','J','P', 'G'); # Important to notice cv2.cv.CV_FOURCC
#fourCC = cv.VideoWriter_fourcc('M','J','P', 'G') #mp4  does not work
#fourCC = cv.VideoWriter_fourcc('M','J','P', 'G') #avi
#fourCC = cv.VideoWriter_fourcc('H','2','6', '4') #Could not get this to work
#fig = plt.figure()
save_path = input_dirs[0]+'/collateral/'
if not os.path.isdir(save_path): os.mkdir(save_path)
for frame_number in range(args.number_frames):

    #for iii in range(40):
    _frame = []
    for ind,c in enumerate(captures):
        ret, frame = c.read()
        if frame is None:
            break
        ## [apply]
    
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(c.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        _frame.append(frame)
        #print(frame_number)
        if args.yolo_boxes:
            for y in yolo[ind][suffix[ind]+'/rgb_'+str(frame_number).zfill(4)+'.png']:
                if y[0] == 'person':
                    (centerX, centerY, width, height) = y[2]
                    #print((int(centerX - (width / 2)), int(centerY - (height / 2))), (int(centerX +(width / 2)),int(centerY + (height / 2))),'oyla_'+str(frame_number).zfill(4)+'.png')
                    cv.rectangle(frame, (int(centerX - (width / 2)), int(centerY - (height / 2))), (int(centerX +(width / 2)),int(centerY + (height / 2))), (0,255,0), 1)
    _frame = np.vstack(_frame)
    
    #print(_frame.shape)
    if frame_number==0:
        size = (_frame.shape[1],_frame.shape[0])
        out = cv.VideoWriter(save_path+"/"+args.output_file_prefix+".mp4", fourCC, 15.0, size, True)
    cv.rectangle(_frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(_frame, str(frame_number), (15, 15),cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    out.write(_frame)
    ## [show]
    
    keyboard = cv.waitKey(300)
    if keyboard == 'q' or keyboard == 27:
        break
    
#ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True,repeat=True)
#plt.show()
#ani.save('/Users/rsingh/Downloads/dynamic_images.mp4')
out.release()
