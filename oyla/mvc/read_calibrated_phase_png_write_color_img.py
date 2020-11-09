##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Version:           2.0 20.10.2019
# Description                                                                                    #
# main  script for  writing files in oyla format;
# RGB in png, depth, amplitude in mat, and depth also in png, xyz pcl
# this scipt will calibrate cameras, filter (if needed) and threshold (min max, x,y if needed). #
# Note that thresholding on amplituden and range should be done at collection time using parametrs.csv#
# This script can also enhance the reflctivity of rgb using oyla reflectivity
##################################################################################################
import numpy as np
import struct
import sys
import os
import cv2
import matplotlib.pyplot as plt
from oyla.mvc.model_read_bin_data import DataRead
from oyla.mvc.utils import  *
from oyla.utils import read_csv_parameters, some_common_utility,  convert_matrix_image, AMBIGUITY_DISTANCE_LUT, MODULATION_FREQUENCIES
import argparse
import json
import distutils.util
import scipy.io
from oyla.mvc.filters import filter_temporal_median
from oyla.enhancement.utils import do_reflectivity_enhancement
import shutil

def read_write_oyla(args):
        input_data_folder_name =  "/".join(args.input_idx_file.split("/")[:-1])
        csv_file = input_data_folder_name+'/parameters.csv'
        parameters = read_csv_parameters(csv_file)
        num_epochs = int(parameters['param']['number_epochs'][0])
        number_chips = len(parameters['param']['chip_id'][0].split(','))
        if int(parameters['param']['number_epochs'][0]) != 1:
                print("ONLY ONE EPOCH LOADING SUPPORTED")
                exit()
        _dim = list(map(int,parameters['adaptive_cmd']['setROI'][0].split(' ')))
        width = _dim[1]-_dim[0]+1
        height = _dim[3]-_dim[2]+1
        if parameters['param']['enableVerticalBinning'][0] == '1':
            height = height // 2
        if parameters['param']['enableHorizontalBinning'][0] == '1':
            width = width //2
        if len(parameters['param']['chip_id'][0].split(',')) == 2:
            width = width *2

        imaging_type = parameters['param']['imaging_type'][0].split('+')
        imaging_mode = parameters['param']['imaging_mode'][0]
        imaging_type = imaging_type[0]
        assert imaging_type == 'Dist_Ampl', "Supports only dist ampl data"

        if 'camera_version' not in parameters['param']:
                assert args.camera_version is not None, "no camera version in parameters file and not specified on command line"
                camera_version = args.camera_version
        else:
                camera_version = parameters['param']['camera_version'][0]
                if 'v2' in camera_version and args.camera_version is not None and 'v3' in args.camera_version:
                        camera_version = args.camera_version
        assert camera_version in CAMERA_VERSION, "Not supported camera version"
        fov_x = FOV[CAMERA_VERSION.index(camera_version)][1]
        fov_y = FOV[CAMERA_VERSION.index(camera_version)][0]
        args.fov_x = fov_x
        args.fov_y = fov_y

        # consistent with controller
        ambiguity_distance, range_max, range_min, saturation_flag, adc_flag, mod_freq, ampl_min,reflectivity_thresh = some_common_utility(parameters,0)

        #if args.reflectivity_thresh is not  None:
        #        reflectivity_thresh = args.reflectivity_thresh
        #range_max = 4500
        #range_min = 500
        #ampl_min = 100




        if not os.path.exists(output_data_folder_name):
                os.makedirs(output_data_folder_name)
         else:
                input(" press key to overwrite oyla output directory")

 

 
        if args.number_images ==0:
                args.number_images = total_num_of_images

        for frame_number in range(0,args.number_images,1):
                print('frame_number',frame_number)
                 if camera_imaging_type =='Dist_Ampl':


                        # AMPLITUDE_SATURATION thresholding should take care of the flags below; esp for pcl and removing data
                        raw_ampl = np.squeeze(frame[:,:,1])
                        raw_phase = np.squeeze(frame)[:,:,0]

                        #if args.rgb:
                        rgb_img,raw_phase,raw_ampl = camera_calibrations(rgb=rgb_data,depth=raw_phase,ampl=raw_ampl,camera_version=camera_version)
                        #if 'oyla_1_camera' in camera_version:
                        height = raw_phase.shape[1]
                        width = raw_phase.shape[0]

                        if 'temporal_median_filter' in filter_params and int(filter_params['temporal_median_filter']):
                                print('Doing temporal median filter')
                                frame_offset = filter_params['temporal_median_filter_size']
                                if frame_number == 0:
                                        median_array = np.zeros((frame_offset+1+frame_offset, raw_phase.shape[0], raw_phase.shape[1],2))
                                        rgb_array = np.zeros((frame_offset+1+frame_offset, rgb_img.shape[0], rgb_img.shape[1],3),dtype = np.uint8)
                                if frame_number < frame_offset+frame_offset+1:
                                        median_array[frame_number, :, :, 0] = raw_phase
                                        median_array[frame_number, :, :, 1] = raw_ampl
                                        rgb_array[frame_number,:,:,:] = rgb_img
                                else:
                                        current_frame = frame_number - frame_offset


                                if current_frame >=frame_offset and (current_frame+frame_offset+1) < total_num_of_images:
                                        if current_frame > frame_offset:
                                                median_array = np.roll(median_array, -1, axis=0)
                                                rgb_array = np.roll(rgb_array, -1, axis = 0)
                                                median_array[frame_offset+frame_offset+0, :, :, 0] = raw_phase
                                                median_array[frame_offset+frame_offset+0, :, :, 1] = raw_ampl
                                                rgb_array[frame_offset+frame_offset+0, :, :, :] = rgb_img
                                        #slice_list = []
                                        if (1+current_frame) > total_num_of_images:
                                                break
                                        #else:
                                        #        slice_list.append(median_array[0:(0+frame_offset+frame_offset)])
                                        results = filter_temporal_median(median_array[0:(0+frame_offset+frame_offset+1)])
                                        raw_phase = results[0]
                                        raw_ampl = results[1]
                                        rgb_img = rgb_array[frame_offset,:,:,:]                  

                        print(rgb_img.shape, rgb_img.dtype)
                        #threshold raw phase and raw amplitude
                        filtered_phase, filtered_ampl, indices = threshold_filter(raw_phase = raw_phase, raw_ampl = raw_ampl, reflectivity_thresh= reflectivity_thresh,
                                                                                     range_max = range_max, range_min = range_min, ampl_min = ampl_min,
                                                                                     filter_params = filter_params, ambiguity_distance = ambiguity_distance,
                                                                                     qp_phase =args.qp_phase, qp_ampl = args.qp_ampl)
                        
                        dist, _ = phase_to_distance(filtered_phase, ambiguity_distance)
                        # outside_range_indices = indices['amplitude_beyond_range']
                        # for k in indices.keys():
                        #         if 'beyond_range' in k and k != 'amplitude_beyond_range':
                        #                 outside_range_indices = (np.concatenate((outside_range_indices[0], indices[k][0])),
                        #                                          np.concatenate((outside_range_indices[1], indices[k][1])))
                        outside_range_indices = indices['filtered_beyond_range']


                        if args.rgb:
                                img = rgb_img.copy()

                                img = cv2.resize(img,(height,width))
                                if args.reflectivity_enhancement:
                                        img = do_reflectivity_enhancement(img, filtered_phase, filtered_ampl, indices)
                                plt.imsave(os.path.join(output_data_folder_name,'rgb_jpg')+'/oyla_'+str(current_frame).zfill(4)+'.jpg',
                                           img)
                                plt.imsave(os.path.join(output_data_folder_name,'rgb_png')+'/oyla_'+str(current_frame).zfill(4)+'.png',
                                           img)
                                plt.imsave(os.path.join(output_data_folder_name,'rgb_jpg_orig')+'/oyla_'+str(current_frame).zfill(4)+'.jpg',
                                           rgb_img)

                        # no_data_indices = indices['amplitude_saturated']
                        # for k in indices.keys():
                        #         if k != 'amplitude_saturated':
                        #                 no_data_indices = (np.concatenate((no_data_indices[0], indices[k][0])),
                        #                                    np.concatenate((no_data_indices[1], indices[k][1])))
                        #         print('rerr',k,no_data_indices[0].shape)
                        # __n = np.ravel_multi_index(no_data_indices,dist.shape)
                        # print('no_data',__n.shape,np.where(filtered_phase==0)[0].shape)
                        for k in indices.keys():
                                print(k,indices[k][0].shape)
                        no_data_indices = outside_range_indices
                        #x, y, z,rcm = transformation3(range_array, args.transform_types, args.fov)
                        x, y, z,rcm = transformation3(dist,args.transform_types,fov_x, fov_y, no_data_indices)
                        
                        x, y, z, rcm, filtered_phase,threshold_indices = threshold_coordinates(x, y, z, rcm, x_max = args.x_max, x_min = args.x_min,
                                                                             y_max = args.y_max, y_min = args.y_min, img = filtered_phase)
                        filtered_ampl[threshold_indices] = 0
                        dist[threshold_indices] = 0
                        #If this is going to be used then we need to take care of threshold_indices by concatenating with outside_range_indices
                        outside_range_indices = (np.concatenate((outside_range_indices[0], threshold_indices[0])),
                                                 np.concatenate((outside_range_indices[1], threshold_indices[1])))
                        
                        img = convert_matrix_image(dist,cmap= 'jet_r', clim_min=range_min, clim_max=range_max,
                                                           saturation_indices = indices['amplitude_saturated'],
                                                           no_data_indices = indices['amplitude_low'],
                                                           outside_range_indices = outside_range_indices)
                        #distmm16 = np.uint16(dist*10)
                        #ampl16 = np.uint16(filtered_ampl)
                        #fp16 = np.uint16(filtered_phase)
                        
                        #scipy.io.savemat(os.path.join(output_data_folder_name,'dist_ampl_mat')+'/oyla_'+str(current_frame).zfill(4)+'.mat',{'dist':dist,'ampl':thresholded_ampl},do_compression=True)
                        #scipy.io.savemat(os.path.join(output_data_folder_name,'dist_ampl_mat_16')+'/oyla_'+str(current_frame).zfill(4)+'.mat',{'dist':distmm16,'ampl':ampl16},do_compression=True)               
                        #plt.imsave(os.path.join(output_data_folder_name,'dist_img_jpg')+'/oyla_'+str(current_frame).zfill(4)+'.jpg',
                        #          img)

                        #cv2.imwrite(os.path.join(output_data_folder_name,'dist_png')+'/oyla_'+str(current_frame).zfill(4)+'.png',distmm16)
                        cv2.imwrite(os.path.join(output_data_folder_name,'phase_png')+'/oyla_'+str(current_frame).zfill(4)+'.png',filtered_phase.astype('uint16'))
                        cv2.imwrite(os.path.join(output_data_folder_name,'ampl_png')+'/oyla_'+str(current_frame).zfill(4)+'.png',filtered_ampl.astype('uint16'))



                        A = np.zeros_like(filtered_phase,dtype=np.uint8)
                        B = np.zeros_like(filtered_phase,dtype=np.uint8)
                        C = np.zeros_like(filtered_phase,dtype=np.uint8)
                        _filtered_ampl = filtered_ampl.copy()
                        _filtered_ampl[_filtered_ampl>511]=511
                        for i in range(filtered_phase.shape[0]):
                                for j in range(filtered_phase.shape[1]):
                                        a = format(filtered_phase[i,j].astype('uint16'), '015b')[::-1]+format(_filtered_ampl[i,j].astype('uint16'), '09b')
                                        A[i,j] = int(a[:8][::-1],2) #LSB of phase
                                        B[i,j] = int(a[8:16][::-1],2) #MSB of phase +MSB of ampl
                                        C[i,j] = int(a[16:],2) #LSB of ampl
                        img = cv2.merge((B,A,C))
                        _img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR_FULL)
                        cv2.imwrite(os.path.join(output_data_folder_name,'phase_ampl_png')+'/oyla_'+str(current_frame).zfill(4)+'.png',_img)
                        # _a = np.c_[x,y,z]
                        # _a = _a.astype('single')

                        # with open(os.path.join(output_data_folder_name,'pcl_'+str(current_frame).zfill(4)+'.xyz'),'w') as fp:
                        #         for __a in _a:
                        #                 fp.write(str(__a[0])+' '+str(__a[1])+' '+str(__a[2])+'\n')


if __name__ == '__main__':
        # note that some of the UI parameters are here. missing range_m* assuming that that comes from parameters.csv
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_idx_file",type = str,required = True)
        parser.add_argument("--transform_types", type = str, default = "cartesian")
        parser.add_argument("--rgb",type=bool, default = True)
        parser.add_argument("--filters_csv",type=str,default = None)


        parser.add_argument("--x_max",type = float, default = None)
        parser.add_argument("--x_min",type = float, default = None)
        parser.add_argument("--y_max",type = float, default = None)
        parser.add_argument("--y_min",type = float, default = None)

        #parser.add_argument("--fov_x", type = int, default = 30)
        #parser.add_argument("--fov_y", type = int, default = 10)
        parser.add_argument("--range_min",type=int,default=None)
        parser.add_argument("--range_max",type=int,default=None)
        parser.add_argument("--ampl_min",type=int,default=None)
        #parser.add_argument('--reflectivity_thresh', type=lambda x:bool(distutils.util.strtobool(x)))

        parser.add_argument('--output_dir_suffix',type = str, default=None)
        parser.add_argument("--camera_version",type = str, default = None)
        parser.add_argument('--no_forward_sync',action="store_true")
        parser.add_argument('--reflectivity_enhancement', action = "store_true")
        parser.add_argument("--number_images",type=int, default = 0)
        parser.add_argument("--qp_phase",type=int, default = None)
        parser.add_argument("--qp_ampl",type=int, default = None)
        args = parser.parse_args()
        print(args.no_forward_sync, args.reflectivity_enhancement)

        read_write_oyla(args)
