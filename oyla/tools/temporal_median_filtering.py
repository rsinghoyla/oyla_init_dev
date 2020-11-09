from __future__ import print_function

import argparse
import numpy
import os
import glob
import sys
import cv2
from timeit import default_timer as timer
from multiprocessing import Process, Pool
from oyla.mvc.model_read_bin_data import DataRead # bin read
from oyla.utils import read_csv_parameters
from oyla.mvc.utils import write_files
import copy
import numpy as np
import shutil
def temporal_median_filter_multi2(model, width, height,  total_num_frames, file_indices, frame_offset=8, simultaneous_frames=1):
    """
    Uses multiprocessing to efficiently calculate a temporal median filter across set of input images.

    DIAGRAM:
    f.o = offset (you actually get 2x what you ask for)
    s.o = simultaneous offset (cuz we do multiple frames at the SAME TIME)
    randframes = we make some random frames for before/after so that we don't run out of frames to use

                   |_____________________total frames______________________|
    randframes_----0      |--f.o----|s.o|---f.o---|


    Args:
        input_data: globbed input directory
        output_dir: path to output
        limit_frames: put a limit on the number of frames
        output_format: select PNG, TIFF, or JPEG (default)
        frame_offset: Number of frames to use for median calculation
        simultaneous_frames: Number of frames to process simultaneously
    Returns:
        str: path to final frames
    """
    start2 = timer()
    


    median_array = numpy.zeros((frame_offset+simultaneous_frames+frame_offset, height, width,2))
    save_path = input_data_folder_name+'/temporal_filter2_'+str(frame_offset)
    if not os.path.isdir(save_path): os.mkdir(save_path)

    # read all the frames into big ol' array
    for frame_number in range(simultaneous_frames+frame_offset+frame_offset):
        print(frame_number)
        np_data, rgb_data = model.mat_to_frame(input_data_folder_name, file_indices[frame_number])
        median_array[frame_number, :, :, :] = np_data[0]
        if frame_number < frame_offset:
            data = []
            data.append(np_data)
            data_f = [0, 1, data, 1,rgb_data]
            key = 'filtered'
            write_files(data_f,key,save_path,fno=frame_number)
            print('a',frame_number)
    #                |_____________________total frames______________________|
    # randframes_----0      |--f.o----|s.o|---f.o---|
    # whole_array = numpy.zeros((total_frames, height, width, 3), numpy.uint8)

    p = Pool(processes=1)
    current_frame = frame_offset
    filtered_array = numpy.zeros((simultaneous_frames, height, width, 2))
    
    
    
    
    while current_frame < total_num_frames:
        if current_frame == frame_offset:
            pass
        else:
            median_array = numpy.roll(median_array, -simultaneous_frames, axis=0)
            for x in range(simultaneous_frames):
                if (current_frame+frame_offset+x) >= total_num_frames:
                    break
                else:
                    np_data,rgb_data = model.mat_to_frame(input_data_folder_name, file_indices[frame_offset+current_frame+x])
                    print('r',frame_offset+current_frame+x)
                    next_array = np_data[0]
                median_array[frame_offset+frame_offset+x, :, :, :] = next_array

                
        slice_list = []
        for x in range(simultaneous_frames):
            if (x+current_frame) > total_num_frames:
                break
            else:
                slice_list.append(median_array[x:(x+frame_offset+frame_offset)])

        # calculate medians in our multiprocessing pool
        results = p.map(median_calc, slice_list)

        for x in range(len(results)):
            filtered_array[x, :, :, 0] = results[x][0]
            filtered_array[x, :, :, 1] = results[x][1]

        np_data,rgb_data = model.mat_to_frame(input_data_folder_name, file_indices[current_frame])
        data = []   
        data.append(np_data)
        data = [0, 1, data, 1,rgb_data]
        data_f = copy.deepcopy(data)
        _tmp = []
        tmp = np.squeeze(filtered_array)
        print(tmp.shape)
        for i,_a in enumerate(data_f[2][0]):
            if i == 0:
                _a = tmp
            _tmp.append(_a)
        data_f[2][0] = np.asarray(_tmp)

        
        key = 'filtered'
        write_files(data_f,key,save_path,fno=current_frame)
        print('b',current_frame)
        current_frame += simultaneous_frames

    for frame_number in range(current_frame, total_num_frames):
        print(fame_number)
        np_data, rgb_data = model.mat_to_frame(input_data_folder_name, file_indices[frame_number])

        data = []
        data.append(np_data)
        data_f = [0, 1, data, 1,rgb_data]
        key = 'filtered'
        write_files(data_f,key,save_path,fno=frame_number)
        print('c',frame_number)
    end2 = timer()
    print("\nTotal Time was: %.02f sec. %.02f sec per frame." % (end2-start2, ((end2-start2)/total_num_frames)))
    return save_path


def median_calc(median_array):
    return numpy.median(median_array[:, :, :, 0], axis=0), \
           numpy.median(median_array[:, :, :, 1], axis=0)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser._optionals.title = 'arguments'

    parser.add_argument("--input_idx_file",type = str,
                        default="../Binary_to_Pointcloud_vispy/BinaryDataDistanceAmplitude/imageDistance.idx")

    args = parser.parse_args()
    
    input_data_folder_name =  "/".join(args.input_idx_file.split("/")[:-1])
    csv_file = input_data_folder_name+'/parameters.csv'
    parameters = read_csv_parameters(csv_file)
    _dim = list(map(int,parameters['adaptive_cmd']['setROI'][0].split(' ')))
    width = _dim[1]-_dim[0]+1
    height = _dim[3]-_dim[2]+1
    if parameters['param']['enableVerticalBinning'][0] == '1':
        height = height // 2
    if parameters['param']['enableHorizontalBinning'][0] == '1':
        width = width //2
    if len(parameters['param']['chip_id'][0].split(',')) == 2:
        width = width *2
    #height = 286
    #width = 200
    args = parser.parse_args()
    total_num_frames = sum(1 for line in open(args.input_idx_file, "r"))
    file_indices  = open(args.input_idx_file).read().splitlines()
    model = DataRead(width, height)
    #width=286
    #height = 200
    save_path = temporal_median_filter_multi2(model, height, width, total_num_frames, file_indices,frame_offset=2)
    shutil.copy(csv_file,save_path+'/parameters.csv')
