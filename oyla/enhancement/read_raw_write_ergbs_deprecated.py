##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Version:           2.0 20.10.2019
# Description                                                                                    #
# main  script for  writing files in kitti format;
# kitti format for 3d is list of (x,y,z,a) tuples in binary format
# a seperate directory of images can also be stored.
# this will convert a raw (depth amplitude) data from oyla into kitti, that is it will also do the spherical to cartesian transformation3
##################################################################################################
import numpy as np
import struct
import sys
import os
import cv2
import matplotlib.pyplot as plt
from oyla.mvc.model_read_bin_data import DataRead
from oyla.mvc.utils import *
import argparse
import json
import distutils.util
from skimage.exposure import equalize_hist
from skimage.segmentation import felzenszwalb
from utils import *
from homofilt import HomomorphicFilter


# note that some of the UI parameters are here. missing range_m* assuming that that comes from parameters.csv
parser = argparse.ArgumentParser()
parser.add_argument("--input_idx_file",type = str,required = True)

parser.add_argument("--transform_types", type = str, default = "cartesian")
#parser.add_argument("--filters_csv",type=str,default = None)
parser.add_argument("--range_min",type=int,default=None)
parser.add_argument("--range_max",type=int,default=None)
parser.add_argument("--ampl_min",type=int,default=50)
#parser.add_argument("--reflectivity_thresh",type=bool,default=True)
parser.add_argument('--reflectivity_thresh', type=lambda x:bool(distutils.util.strtobool(x)))
parser.add_argument("--enhancement",type = str,required = True)
parser.add_argument('--eq_hist',action="store_true")
parser.add_argument('--forward_sync',action="store_true")
parser.add_argument("--camera_version",type = str, default = None)
parser.add_argument("--no_resize",action="store_true")
args = parser.parse_args()
print(args.reflectivity_thresh)

input_data_folder_name =  "/".join(args.input_idx_file.split("/")[:-1])
csv_file = input_data_folder_name+'/parameters.csv'
parameters = read_csv_parameters(csv_file)
num_epochs = int(parameters['param']['number_epochs'][0])
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

# consistent with controller
ambiguity_distance, range_max, range_min, saturation_flag, adc_flag, mod_freq, ampl_min,reflectivity_thresh = some_common_utility(parameters,0)
if args.range_min is not None:
        range_min = args.range_min
if args.range_max is not None:
        range_max = args.range_max
if args.ampl_min is not None:
        ampl_min = args.ampl_min
if args.reflectivity_thresh is not  None:
        reflectivity_thresh = args.reflectivity_thresh

filter_params = {}
# if args.filters_csv is not None and (imaging_type == 'Dist_Ampl'):
#         filter_parameters = read_csv_parameters("filter_params.csv")
#         for k in filter_parameters['filter_cmd'].keys():
#                 #print(parameters['filter_cmd'][k][self.epoch_number])
#                 try:
#                         filter_params[k] = int(filter_parameters['filter_cmd'][k][0])
#                         parameters['filter_cmd'][k][0] = filter_parameters['filter_cmd'][k][0]
#                 except ValueError:
#                         pass
#         print('FILTERS',filter_params)
    
total_num_of_images = sum(1 for line in open(args.input_idx_file, "r"))
file_indices  = open(args.input_idx_file).read().splitlines() 

output_data_folder_name = os.path.join(input_data_folder_name,'kitti')
if not os.path.exists(output_data_folder_name):
        os.makedirs(output_data_folder_name)
        os.makedirs(os.path.join(output_data_folder_name,'2d'))

string = 'ergb'
if args.eq_hist:
        string += '_eqh'
if args.forward_sync:
        string += '_fs'
if args.no_resize:
        string += '_no_resize'
if 'v3' in camera_version:
        string += '_'+camera_version.split('_')[-1]
string += '_'

if not os.path.exists(os.path.join(os.path.join(output_data_folder_name,'2d'),string+args.enhancement)):
        os.makedirs(os.path.join(os.path.join(output_data_folder_name,'2d'),string+args.enhancement))
output_data_folder_name = os.path.join(os.path.join(output_data_folder_name,'2d'),string+args.enhancement) 
with open(output_data_folder_name+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
model = DataRead(width,height)
homo_filter = HomomorphicFilter()
for frame_number in range(total_num_of_images):
        np_data = model.bin_to_frame(input_data_folder_name,file_indices[frame_number],frame_number,imaging_type=imaging_type)
        if np_data is not None:
                # legacy, bin depth, mat rgb
                _,rgb_data = model.mat_to_frame(input_data_folder_name,file_indices[frame_number])
        elif np_data is None:
                # going forward
                np_data, rgb_data = model.mat_to_frame(input_data_folder_name,file_indices[frame_number])
                if args.forward_sync and frame_number != total_num_of_images-1:
                        _, rgb_data = model.mat_to_frame(input_data_folder_name,file_indices[frame_number+1])
                        
        camera_data = np_data

        frame = camera_data[0]
        camera_chip_ip = camera_data[1]
        _frame_number = camera_data[2]
        camera_imaging_type = camera_data[3]
        if camera_imaging_type =='Dist_Ampl':
                

                # AMPLITUDE_SATURATION thresholding should take care of the flags below; esp for pcl and removing data
                raw_ampl = np.squeeze(frame[:,:,1])
                raw_phase = np.squeeze(frame)[:,:,0]
                rgb_img,raw_phase,raw_ampl = camera_calibrations(rgb=rgb_data, depth=raw_phase, ampl=raw_ampl,camera_version=camera_version)
                #if 'oyla_1_camera' in camera_version:
                height = raw_phase.shape[1]
                width = raw_phase.shape[0]

                # if you need to hack the raw phase to be displayed along with filtered_phase
                
                #threshold raw phase and raw amplitude
                filtered_phase, thresholded_ampl, indices = threshold_filter(raw_phase = raw_phase, raw_ampl = raw_ampl, reflectivity_thresh= reflectivity_thresh,
                                                                             range_max = range_max, range_min = range_min, ampl_min = ampl_min,
                                                                             filter_params = filter_params, ambiguity_distance = ambiguity_distance)

        
                # if camera_version == 'oyla_2_camera':
                #     filtered_phase = cv2.transpose(filtered_phase)
                #     filtered_ampl = cv2.transpose(filtered_ampl)
                # if camera_version == 'oyla_1_camera_v0':
                #     filtered_phase = np.rot90(filtered_phase)  # there is a difference to transpose and rotate mainly that it roatets the other way round
                #     filtered_ampl = np.rot90(thresholded_ampl)

                #rgb_img = rgb_img
                if not args.no_resize:
                        rgb_img = cv2.resize(rgb_img,(height,width))
                # if camera_version == 'oyla_1_camera_v0':
                #     rgb_img = np.fliplr(rgb_img)
                rgb_img = rgb_img.astype('float32')/255.0
                hsv_img = cv2.cvtColor(rgb_img,cv2.COLOR_RGB2HSV)
                ehsv_img = hsv_img.copy()
                #f = filtered_phase/np.max(filtered_phase)
                _thresholded_ampl = thresholded_ampl.astype('float32')/np.max(thresholded_ampl)
                _filtered_phase = filtered_phase.astype('float32')/np.max(filtered_phase)

                hsv_flag = True
              
                # if 'reflectivity' in args.enhancement:
                #     rgb_reflectivity = plt.imread(input_data_folder_name,+'/kitti/2d/intrinsic/oyla_'+str(frame_number)+'-r.png')
                #     gray_shading = plt.imread(input_data_folder_name,+'/kitti/2d/intrinsic/oyla_'+str(frame_number)+'-s.png')
                #     depth_reflectivity = 0*thresholded_ampl.astype('float32')
                #     for i in range(thresholded_ampl.shape[0]):
                #         for j in range(thresholded_ampl.shape[1]):
                #             a = thresholded_ampl[i,j]
                #             p = filtered_phase[i,j]
                #             if a>0:
                #                 depth_reflectivity[i,j] = fit_reflectivity(a,p)
                #     depth_reflectivity = depth_reflectivity.astype('float32')/np.max(depth_reflectivity)
                #     rgb_reflectivity = cv2.transpose(rgb_reflectivity)
                #     gray_shading = gray_shading.transpose()
                #     if args.enhancement == 'reflectivity_shading':
                #         shading = np.where(gray_shading>depth_reflectivity,gray_shading,depth_reflectivity)
                #         ergb = rgb_reflectivity*shading[:,:,np.newaxis]
                #     elif args.enhancement == 'reflectivity_reflectivity':
                #         hsv_reflectivity = cv2.cvtColor(rgb_reflectivity,cv2.COLOR_RGB2HSV)
                #         h,s,v = cv2.split(hsv_reflectivity)
                #         v[np.where(depth_reflectivity>v)] = depth_reflectivity[np.where(depth_reflectivity>v)]
                #         e_hsv_reflectivity = cv2.merge((h,s,v))
                #         e_rgb_reflectivity = cv2.cvtColor(e_hsv_reflectivity,cv2.COLOR_HSV2RGB)
                #         ergb = e_rgb_reflectivity*gray_shading[:,:,np.newaxis]
                #     hsv_flag = False
                # if args.enhancement=='reflectivity_mult':
                #         oyla_reflectivity = fit_reflectivity(thresholded_ampl, filtered_phase, camera_version=camera_version)
                #         oyla_reflectivity[oyla_reflectivity>1]=1
                #         ind = np.where(oyla_reflectivity>0)
                #         s = np.max(rgb_img,axis=2)
                #         ergb_img = rgb_img.copy()
                        
                #         ergb_img[ind] = ergb_img[ind]*(oyla_reflectivity[ind]/(s[ind]+0.000001))[:,np.newaxis]
                #         hsv_flag = False
                #         #print(rgb_img.shape,np.max(ergb_img));
                if 'reflectivity_comp' in args.enhancement:
                        oyla_reflectivity = fit_reflectivity(thresholded_ampl, filtered_phase, camera_version=camera_version)
                        h,s,v = cv2.split(hsv_img)
                        
                        if '_cutoff_1_' in args.enhancement:
                                illum,refl = homo_filter.get_illumination_reflectance(I=v,filter_params=[1,2])
                        elif '_cutoff_100_' in args.enhancement:
                                illum,refl = homo_filter.get_illumination_reflectance(I=v,filter_params=[100,2])
                        else:
                                illum,refl = homo_filter.get_illumination_reflectance(I=v,filter_params=[30,2])
                        if args.no_resize:
                                refl = cv2.resize(refl,(height,width))
                        scale = np.max(refl)
                        # if args.enhancement == 'reflectivity_comp_th_0':
                        #         oyla_reflectivity[oyla_reflectivity>1]=0
                        #         print('reflectivity_comp_th_0',frame_number)

                        if  '_supress_saturation_' in args.enhancement:
                                print('reflectivity_comp_supress_saturatio',frame_number)
                                oyla_reflectivity = supress_saturation(oyla_reflectivity, indices)
                                
                        # if '_th_1_' in args.enhancement:
                        #         oyla_reflectivity[oyla_reflectivity>1]=1
                        #         print('reflectivity_comp_th_1',frame_number)
                        # elif '_th_0_1_' in args.enhancement:
                        #         print('reflectivity_comp_th_0_1',frame_number)

                        oyla_reflectivity[oyla_reflectivity>2] = 0
                        oyla_reflectivity[oyla_reflectivity>1] = 1
                        
                        # elif args.enhancement == 'reflectivity_comp_clean': 
                        #         oyla_reflectivity = clean_algorithm(oyla_reflectivity)
                        # elif args.enhancement == 'reflectivity_comp_clean2':
                        #         oyla_reflectivity = clean_algorithm2(oyla_reflectivity)
                        # elif  'reflectivity_comp_supress_saturation_nom' in args.enhancement:
                        #         print('reflectivity_comp_supress_saturation_nom',frame_number)
                        #         oyla_reflectivity = supress_saturation(oyla_reflectivity, indices)
                        #         oyla_reflectivity[oyla_reflectivity>np.max(refl)] = 0
                        #         m = 1
                        if '_highpass_reflectivity_' in args.enhancement:
                                _,oyla_reflectivity = homo_filter.get_illumination_reflectance(I=oyla_reflectivity,filter_params=[1,2])
                                scale = 1

                        if '_minmax_scaling_' in args.enhancement:
                                print('doing minmax')
                                mi = np.min(refl)
                                ma = np.max(refl)
                                refl = (refl-mi)/(ma-mi)
                                scale = 1
                                
                        ind = np.where((refl/scale<(oyla_reflectivity)))
                        refl[ind] = (oyla_reflectivity)[ind]*scale
                        #print(ind[0].shape,np.mean(refl[ind]))
                        if '_minmax_scaling' in args.enhancement:
                                refl = refl*(ma-mi)+mi
                        if args.no_resize:
                                refl = cv2.resize(refl,(illum.shape[1],illum.shape[0]))
                        _v = illum*refl-1
                        _v = _v.astype('float32')
                        _v[_v>1]=1
                        ehsv_img = cv2.merge((h,s,_v))
                        
                        #print(rgb_img.shape,np.max(ergb_img))
                elif 'stransform' in args.enhancement:
                    h,s,v = cv2.split(hsv_img)
                    _v = v.copy()
                    #for _ in range(3):
                    #    _thresholded_ampl = cv2.medianBlur(_thresholded_ampl, 3)
                    if 'scalar' in args.enhancement:
                        __v = Stransform(_v)
                    else:
                        __v = Stransform(_v,delta2=1.0,n=3,m=_thresholded_ampl)
                    ehsv_img = cv2.merge((h,s,__v))
                elif args.enhancement == 'iso_depth_eq_hist':
                    h,s,v = cv2.split(hsv_img)
                    image_felzenszwalb = felzenszwalb(_filtered_phase,multichannel=False,min_size=50,sigma=1.5,scale=500)
                    _v = np.zeros_like(v)
                    for label in np.unique(image_felzenszwalb):
                        mask = np.zeros_like(image_felzenszwalb)
                        #ind = np.where((image_felzenszwalb==label)&(_filtered_phase>0))
                        ind =  np.where((image_felzenszwalb==label))
                        mask[ind] = 1
                        #print(np.sum(mask))
                        __v = equalize_hist(v,mask=mask)
                        _v[ind] = __v[ind]
                    ehsv_img = cv2.merge((h,s,_v))
                #else:
                #        print("error in enhancement method")
                #        exit(0)
                if hsv_flag:
                    ergb_img = cv2.cvtColor(ehsv_img,cv2.COLOR_HSV2RGB)
                    ergb_img[ergb_img>1] = 1
                    ergb_img[ergb_img<0] = 0
                    #print('came')
                if args.eq_hist:
                    ergb_img = rgb_equalize_histogram(ergb_img)
                    
                plt.imsave(os.path.join(output_data_folder_name,'oyla_'+str(frame_number).zfill(4)+'.png'),
                           (ergb_img))
                        
        
