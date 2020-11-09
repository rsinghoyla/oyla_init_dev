import numpy as np
import cv2 as cv
import argparse
import os
parser = argparse.ArgumentParser(description='This program can be used to decode a video into frames')
parser.add_argument('--input_video_file',type=str,required = True)
parser.add_argument('--output_dir_suffix',type=str,default='')
args = parser.parse_args()
output_data_folder_name = '/'.join(args.input_video_file.split('/')[:-1])
print(output_data_folder_name)
if not os.path.exists(os.path.join(output_data_folder_name,'decoded_phase_png'+args.output_dir_suffix)):
    #os.makedirs(os.path.join(output_data_folder_name,'decoded_jpg'+args.output_dir_suffix))
    os.makedirs(os.path.join(output_data_folder_name,'decoded_phase_png'+args.output_dir_suffix))
    os.makedirs(os.path.join(output_data_folder_name,'decoded_ampl_png'+args.output_dir_suffix))
cap = cv.VideoCapture(args.input_video_file)
current_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #
    dec_hsv = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    h,s,v = cv.split(dec_hsv)
    decoded_phase = np.zeros_like(h,dtype=np.uint16)
    decoded_ampl = np.zeros_like(s,dtype=np.uint16)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            c = format(v[i,j],'08b')[::-1]+format(h[i,j],'08b')[::-1]+format(s[i,j],'08b')
            decoded_phase[i,j] = int(c[:16][::-1],2)
            decoded_ampl[i,j] = int(c[16:],2)
    cv.imwrite(os.path.join(output_data_folder_name,'decoded_phase_png'+args.output_dir_suffix)+'/oyla_'+str(current_frame).zfill(4)+'.png',decoded_phase.astype('uint16'))
    cv.imwrite(os.path.join(output_data_folder_name,'decoded_ampl_png'+args.output_dir_suffix)+'/oyla_'+str(current_frame).zfill(4)+'.png',decoded_ampl.astype('uint16'))
    current_frame += 1
cap.release()
