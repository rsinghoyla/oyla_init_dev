
import cv2 as cv
import argparse
import os
parser = argparse.ArgumentParser(description='This program can be used to decode a video into frames')
parser.add_argument('--input_video_file',type=str,required = True)
parser.add_argument('--output_dir_suffix',type=str,default='')
args = parser.parse_args()
output_data_folder_name = '/'.join(args.input_video_file.split('/')[:-1])
print(output_data_folder_name)
if not os.path.exists(os.path.join(output_data_folder_name,'decoded_png'+args.output_dir_suffix)):
    #os.makedirs(os.path.join(output_data_folder_name,'decoded_jpg'+args.output_dir_suffix))
    os.makedirs(os.path.join(output_data_folder_name,'decoded_png'+args.output_dir_suffix))
cap = cv.VideoCapture(args.input_video_file)
current_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    cv.imwrite(os.path.join(output_data_folder_name,'decoded_png'+args.output_dir_suffix)+'/oyla_'+str(current_frame).zfill(4)+'.png',frame)
    #cv.imwrite(os.path.join(output_data_folder_name,'decoded_jpg')+'/oyla_'+str(current_frame).zfill(4)+'.jpg',frame)
    current_frame += 1
cap.release()
