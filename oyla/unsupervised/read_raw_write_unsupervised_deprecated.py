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
import open3d as o3d
import scipy.io
from utils import *



# note that some of the UI parameters are here. missing range_m* assuming that that comes from parameters.csv
parser = argparse.ArgumentParser()
parser.add_argument("--input_idx_file",type = str,required = True)
parser.add_argument("--transform_types", type = str, default = "cartesian")
#parser.add_argument("--fov_x", type = int, default = 30)
#parser.add_argument("--fov_y", type = int, default = 10)
parser.add_argument("--x_max",type = float, default = None)
parser.add_argument("--x_min",type = float, default = None)
parser.add_argument("--y_max",type = float, default = None)
parser.add_argument("--y_min",type = float, default = None)
#parser.add_argument("--filters_csv",type=str,default = None)
parser.add_argument("--range_min",type=int,default=None)
parser.add_argument("--range_max",type=int,default=None)
parser.add_argument("--ampl_min",type=int,default=50)
parser.add_argument('--reflectivity_thresh', type=lambda x:bool(distutils.util.strtobool(x)))
parser.add_argument("--camera_version",type = str, default = None)
parser.add_argument("--bgnd_sub_algo",type = str, default ="MOG2")
parser.add_argument("--output_prefix",type=str,default="Test")
parser.add_argument("--number_images",type=int, default = 0)
parser.add_argument("--segmentation_sigma",type=float, default = 1.5)
parser.add_argument("--segmentation_scale",type=float, default = 500)
parser.add_argument("--segmentation_min_size",type=int, default = 50)
parser.add_argument("--segmentation_eps",type=int, default = 100)
parser.add_argument("--ground_plane_removal",type = bool, default = False)
parser.add_argument("--view",type=str, default='x')
parser.add_argument("--depth",type=lambda x:bool(distutils.util.strtobool(x)),default = True)
parser.add_argument("--rgb",type=lambda x:bool(distutils.util.strtobool(x)),default = True)
parser.add_argument("--pcd_segment",type=lambda x:bool(distutils.util.strtobool(x)),default = True)
parser.add_argument("--felz_segment",type=lambda x:bool(distutils.util.strtobool(x)),default = False)
#parser.add_argument("--pcd_segs_stransform",type=lambda x:bool(distutils.util.strtobool(x)),default = False)
parser.add_argument("--mask",type=lambda x:bool(distutils.util.strtobool(x)),default = True)
args = parser.parse_args()

segmentation_scale = args.segmentation_scale
segmentation_sigma = args.segmentation_sigma
segmentation_min_size = args.segmentation_min_size
segmentation_eps = args.segmentation_eps

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

#fourCC = cv2.VideoWriter_fourcc('M','J','P', 'G'); # Important to notice cv2.cv.CV_FOURCC
fourCC = cv2.VideoWriter_fourcc(*'MP4V')    
model = DataRead(width,height)

#create Background Subtractor objects
if args.bgnd_sub_algo == 'MOG2':
    dist_backSub = cv2.createBackgroundSubtractorMOG2()
    rgb_backSub = cv2.createBackgroundSubtractorMOG2()
else:
    dist_backSub = cv2.createBackgroundSubtractorKNN()
    rgb_backSub = cv2.createBackgroundSubtractorKNN()

counts = {}
counts['mask_mismatch'] = [] 
open3d_clusters_size = {}

if args.number_images ==0:
        args.number_images = total_num_of_images

save_path = input_data_folder_name+'/collateral'
if not os.path.isdir(save_path): os.mkdir(save_path)

pcd = o3d.geometry.PointCloud()
for frame_number in range(args.number_images):
        print("\nNew frame",frame_number)
        np_data = model.bin_to_frame(input_data_folder_name,file_indices[frame_number],frame_number,imaging_type=imaging_type)
        if np_data is not None and args.rgb:
                # legacy, bin depth, mat rgb
                _,rgb_data = model.mat_to_frame(input_data_folder_name,file_indices[frame_number])
        elif np_data is None:
                # going forward
                np_data, rgb_data = model.mat_to_frame(input_data_folder_name,file_indices[frame_number])
        #elif not args.rgb:
                # always if rgb is not available
        #        rgb_data = np.zeros((640,480))
        camera_data = np_data

        data = camera_data[0]
        camera_chip_ip = camera_data[1]
        _frame_number = camera_data[2]
        camera_imaging_type = camera_data[3]
        if camera_imaging_type =='Dist_Ampl':

                frame = None
                # AMPLITUDE_SATURATION thresholding should take care of the flags below; esp for pcl and removing data
                raw_ampl = np.squeeze(data[:,:,1])
                raw_phase = np.squeeze(data)[:,:,0]
                
                #if args.rgb:
                rgb_img,raw_phase,raw_ampl = camera_calibrations(rgb=rgb_data,depth=raw_phase,ampl=raw_ampl,camera_version=camera_version)
                #if 'oyla_1_camera' in camera_version:
                height = raw_phase.shape[1]
                width = raw_phase.shape[0]
                
                #threshold raw phase and raw amplitude
                filtered_phase, thresholded_ampl, indices = threshold_filter(raw_phase = raw_phase, raw_ampl = raw_ampl, reflectivity_thresh= reflectivity_thresh,
                                                                             range_max = range_max, range_min = range_min, ampl_min = ampl_min,
                                                                             filter_params = filter_params, ambiguity_distance = ambiguity_distance)

                dist, _ = phase_to_distance(filtered_phase, ambiguity_distance)
                # for pcd construction
                no_data_indices = indices['amplitude_saturated']
                for k in indices.keys():
                        if k != 'amplitude_saturated':
                                no_data_indices = (np.concatenate((no_data_indices[0], indices[k][0])),
                                                   np.concatenate((no_data_indices[1], indices[k][1])))

                x, y, z, rcm = transformation3(dist,args.transform_types,fov_x, fov_y, no_data_indices)
                x, y, z, rcm, dist = threshold_coordinates(x, y, z, rcm, x_max = args.x_max, x_min = args.x_min,
                                                           y_max = args.y_max, y_min = args.y_min,img=dist)
                
                pcd.points = o3d.utility.Vector3dVector(np.vstack((x,y,z)).transpose())
                # NMlabels = np.array(pcd.cluster_dbscan(eps=segmentation_eps, min_points=segmentation_min_size, print_progress=False))
                # NMmax_label = NMlabels.max()
                # print('Open3D  unique labels with counts',np.unique(NMlabels,return_counts=True))
                # plt.imshow(dist)
                # plt.pause(0.00001)
                outside_range_indices = indices['amplitude_beyond_range']
                for k in indices.keys():
                        if ('beyond_range' in k and k != 'amplitude_beyond_range') or 'ground_plane' in k:
                                outside_range_indices = (np.concatenate((outside_range_indices[0], indices[k][0])),
                                                         np.concatenate((outside_range_indices[1], indices[k][1])))
                
                dist_img = convert_matrix_image(dist,cmap= 'jet_r', clim_min=range_min, clim_max=range_max,
                                                   saturation_indices = indices['amplitude_saturated'],
                                                   no_data_indices = indices['amplitude_low'],
                                                   outside_range_indices = outside_range_indices)[:,:,:3]
                #get the foreground mask
                if args.mask:
                        dist_fgMask = dist_backSub.apply(dist_img)
                else:
                        dist_fgMask = np.ones_like(dist)*255
                #get the masked distance image
                masked_dist = np.zeros_like(dist)
                masked_dist[dist_fgMask==255] = dist[dist_fgMask==255]
                                
                non_mask_indices = np.where(dist_fgMask != 255)
                mask_indices = np.where(dist_fgMask == 255)
                
                masked_dist_img = convert_matrix_image(masked_dist,cmap= 'jet_r', clim_min=range_min, clim_max=range_max,
                                                       no_data_indices = non_mask_indices)[:,:,:3]
                if args.depth:
                        if args.mask:
                                frame = np.hstack((dist_img,masked_dist_img))
                        else:
                                frame = dist_img
                                
                #get the foreground mask on rgb image and get masked rgb image
                rgb_img = cv2.resize(rgb_img,(height,width))
                if args.mask:
                        rgb_fgMask = rgb_backSub.apply(rgb_img)
                else:
                        rgb_fgMask = np.ones((width,height))*255
                        
                masked_rgb_img = np.zeros_like(rgb_img)
                masked_rgb_img[rgb_fgMask==255] = rgb_img[rgb_fgMask==255]
                counts['mask_mismatch'].append(np.where(rgb_fgMask != dist_fgMask)[0].shape)
                
                #frame for video being stacked together.
                if args.mask:
                        _frame = np.hstack((rgb_img,masked_rgb_img))
                else:
                        _frame = rgb_img
                        
                if args.depth and args.rgb:
                    frame = np.vstack((_frame,frame))
                elif args.rgb:
                    frame = _frame
                
                # this is needed for cv2 conversions etc and also affects segmentation
                if args.felz_segment:
                        felz_segs_dist_img, counts = wrapper_around_segmentation(dist,'dist',min_size=segmentation_min_size,sigma=segmentation_sigma,
                                                                                 scale = segmentation_scale,counts = counts)
                        felz_segs_masked_dist_img, counts = wrapper_around_segmentation(masked_dist,'masked_dist',min_size=segmentation_min_size,
                                                                                        sigma=segmentation_sigma,scale = segmentation_scale,counts = counts)
                
                                        
                        #felz_segs = felzenszwalb(rgb_fgMask,multichannel=False,min_size=40,sigma=0.4,scale = 10)
                        #print('Mask RGB Number of segments',np.unique(felz_segs).shape[0])
                        felz_segs_rgb_img,counts = wrapper_around_segmentation(rgb_img,'rgb',min_size=segmentation_min_size,sigma=segmentation_sigma,
                                                                               scale = segmentation_scale,counts = counts)
                        felz_segs_masked_rgb_img,counts = wrapper_around_segmentation(masked_rgb_img,'masked_rgb',min_size=segmentation_min_size,
                                                                                      sigma=segmentation_sigma, scale = segmentation_scale,counts = counts)
                        
                #all this is for converting to rgb to add in video
                if args.felz_segment and args.rgb and args.depth:
                    _frame = np.vstack((felz_segs_masked_rgb_img,felz_segs_masked_dist_img))
                elif args.felz_segment and args.rgb:
                    _frame = felz_segs_masked_rgb_img
                elif args.felz_segment and args.depth:
                    _frame = felz_segs_masked_dist_img

                if args.felz_segment:
                    frame = np.hstack((frame,_frame))
                    
                
                # labels = np.array(pcd.cluster_dbscan(eps=50, min_points=20, print_progress=True))
                # max_label = labels.max()
                # print('Open3D unique labels with counts',np.unique(labels,return_counts=True))
                # if 'dist_o3d_segs' not in counts:
                #         counts['dist_o3d_segs'] = []
                # counts['dist_o3d_segs'].append(np.unique(labels).shape[0])

                # for bev we will do y,z
                y_max = range_max*np.sin(fov_y/360*np.pi)+height/2
                y_min = -y_max
                z_max = range_max
                z_min = 0
                x_max = range_max*np.sin(fov_x/360*np.pi)+width/2
                x_min = -x_max

                if args.view == 'xz':
                        bev,_,_ = depth_to_xyz_view_img(pcd,view='xz',a_max = x_max, a_min = x_min, b_max = z_max, b_min = z_min)
                elif args.view == 'yz':
                        bev,_,_ = depth_to_xyz_view_img(pcd,view='yz',a_max = y_max, a_min = y_min, b_max = z_max, b_min = z_min)
                elif args.view == 'xy':
                        bev,_,_ = depth_to_xyz_view_img(pcd,view='xy',a_max = x_max, a_min = x_min, b_max = y_max, b_min = y_min)
                else:
                    bev = np.zeros_like(dist)
                    args.view = False
                bev = np.rot90(bev)
                #plt.imshow(np.rot90(bev))
                #plt.pause(0.000001)
                if args.felz_segment:
                        felz_segs_bev_img,counts = wrapper_around_segmentation(bev,'bev',min_size=segmentation_min_size,
                                                                               sigma=segmentation_sigma, scale = segmentation_scale,counts = counts)
                else:
                        felz_segs_bev_img = np.zeros_like(bev)

                
                dist_flatten = dist.flatten()
                indices_dist_flatten = np.where(dist_flatten>0)[0]
                masked_indices_dist_flatten = np.ravel_multi_index(mask_indices,dist.shape)
                #print(np.setdiff1d(mask_indices_dist_flatten,indices_dist_flatten).shape,mask_indices_dist_flatten.shape,indices_dist_flatten.shape)
                #print(masked_dist_flatten[np.setdiff1d(mask_indices_dist_flatten,indices_dist_flatten)])
                _,_,_masked_indices = np.intersect1d(masked_indices_dist_flatten,indices_dist_flatten,return_indices=True)
                masked_pcd = pcd.select_by_index(_masked_indices)
                # no_data_indices = (np.concatenate((no_data_indices[0], non_mask_indices[0])),
                #                                    np.concatenate((no_data_indices[1], non_mask_indices[1])))

                # x, y, z,rcm = transformation3(dist,args.transform_types,fov_x, fov_y, no_data_indices)
                
                # x, y, z, rcm = threshold_coordinates(x, y, z, rcm, x_max = args.x_max, x_min = args.x_min,
                #                                      y_max = args.y_max, y_min = args.y_min)
                #assumption is that the this is the ground plane, but there are actually many planes, so we have to iterate and we dont know where to stop
                
                        
                masked_bev = np.zeros_like(bev)
                pcd_segs = np.zeros_like(bev)
                felz_segs_masked_bev_img = np.zeros_like(felz_segs_bev_img)
                labels = -1
                if len(_masked_indices)>0:
                        if args.ground_plane_removal:
                                print("doing explicit ground_plane removal but its adhoc still")
                                _ind = []
                                for _ in range(1):
                                        plane_model, _gp_indices = masked_pcd.segment_plane(distance_threshold=2, ransac_n=3, num_iterations=100)
                                        #print(plane_model)
                                        #if plane_model[0] > 10e-01 or plane_model[0] < 9e-01:
                                        #        break
                                        masked_dist_flatten = masked_dist.flatten()
                                        indices_dist_flatten = np.where(masked_dist_flatten>0)[0]
                                        masked_dist_flatten[indices_dist_flatten[_gp_indices]] = 0
                                        masked_dist = masked_dist_flatten.reshape(dist.shape)
                                        _ind.extend(indices_dist_flatten[_gp_indices].tolist())
                                        masked_pcd = masked_pcd.select_by_index(_gp_indices, invert=True)
                                #this will be needed if we make a masked_dist_img by using convert matrix image see
                                indices['ground_plane'] = np.unravel_index(_ind,dist.shape)

                        if args.pcd_segment:
                                #if frame_number == 199:
                                #        o3d.io.write_point_cloud('/Users/rsingh/Downloads/test.pcd',masked_pcd)
                                labels = np.array(masked_pcd.cluster_dbscan(eps=segmentation_eps, min_points=segmentation_min_size, print_progress=False))
                                max_label = labels.max()
                                print('Open3D Masked unique labels with counts',np.unique(labels,return_counts=True))
                                open3d_clusters_size[str(frame_number)] = np.unique(labels,return_counts=True)
                                xyz = np.asarray(masked_pcd.points)
                                _t = np.c_[xyz,labels]
                                masked_dist_flat = masked_dist.flatten()
                                masked_indices_dist_flatten = np.where(masked_dist.flat>0)[0]
                                #print(masked_indices_dist_flatten.shape,masked_pcd)
                                _mE = []
                                for l in np.unique(labels):
                                        #print(l,labels==l,indices_nonzero_flat)
                                        masked_dist_flat[masked_indices_dist_flatten[labels==l]] = l+1
                                        _mE.extend(np.mean(_t[np.where(_t[:,3]==l),:3],axis=1))
                                open3d_clusters_size[str(frame_number)] = np.vstack((open3d_clusters_size[str(frame_number)][0].transpose(),open3d_clusters_size[str(frame_number)][1].transpose(),np.asarray(_mE).transpose()))
                                #masked_dist_flat[indices_nonzero_flat[0][_inliers]] = 0
                                pcd_segs = masked_dist_flat.reshape(masked_dist.shape)
                        
                        #masked_bev = depth_to_xyz_view_img(masked_pcd,view='xz', a_max = x_max, a_min = x_min, b_max = z_max, b_min = z_min)
                        if args.view == 'xz':
                                masked_bev,_,_ = depth_to_xyz_view_img(masked_pcd,view='xz',a_max = x_max, a_min = x_min, b_max = z_max, b_min = z_min)
                        elif args.view == 'yz':
                                masked_bev,_,_ = depth_to_xyz_view_img(masked_pcd,view='yz',a_max = y_max, a_min = y_min, b_max = z_max, b_min = z_min)
                        elif args.view == 'xy':
                                masked_bev,_,_ = depth_to_xyz_view_img(masked_pcd,view='xy',a_max = x_max, a_min = x_min, b_max = y_max, b_min = y_min)
                        masked_bev = np.rot90(masked_bev)
                        if args.felz_segment:
                                felz_segs_masked_bev_img,counts = wrapper_around_segmentation(masked_bev,'masked_bev',min_size=segmentation_min_size,
                                                                                              sigma=segmentation_sigma, scale = segmentation_scale,counts=counts)
 
                        #plt.imshow(felz_segs)
                        #plt.pause(0.000001)
                       
 
                else:
                        if 'masked_bev' not in counts:
                                counts['masked_bev'] = []
                        counts['masked_bev'].append(0)
                        
                if 'masked_dist_o3d' not in counts:
                        counts['masked_dist_o3d'] = []
                counts['masked_dist_o3d'].append(np.unique(labels).shape[0])
                
                _t = bev.astype('float32')/np.max(bev+0.000001)
                bev_img = cv2.cvtColor((_t*255).astype('uint8'),cv2.COLOR_GRAY2RGB)
                _t = masked_bev.astype('float32')/np.max(masked_bev+0.000001)
                masked_bev_img = cv2.cvtColor((_t*255).astype('uint8'),cv2.COLOR_GRAY2RGB)
                _t = pcd_segs.astype('float32')/np.max(pcd_segs+0.00001)
                pcd_segs_img = cv2.cvtColor((_t*255).astype('uint8'),cv2.COLOR_GRAY2RGB)

                
                pcd_segs_rgb_img = np.zeros_like(rgb_img)
                pcd_segs_thresholded_ampl = np.zeros_like(thresholded_ampl)
                #print(pcd_segs.shape,rgb_img.shape,pcd_segs_img.shape)
                if pcd_segs.shape[0] == rgb_img.shape[0]:
                    pcd_segs_rgb_img[pcd_segs>0] = rgb_img[pcd_segs>0]
                    pcd_segs_thresholded_ampl[pcd_segs>0] = thresholded_ampl[pcd_segs>0]
                # if args.pcd_segs_stransform:
                #     pcd_segs_rgb_img = pcd_segs_rgb_img.astype('float32')/255.0
                #     hsv_img = cv2.cvtColor(pcd_segs_rgb_img,cv2.COLOR_RGB2HSV)
                #     _thresholded_ampl = pcd_segs_thresholded_ampl.astype('float32')/np.max(pcd_segs_thresholded_ampl)
                #     h,s,v = cv2.split(hsv_img)
                #     _v = v.copy()
                #     __v = Stransform(_v,delta2=1.0,m=_thresholded_ampl,n=3)
                #     hsv_img = cv2.merge((h,s,__v))
                #     pcd_segs_rgb_img = cv2.cvtColor(hsv_img,cv2.COLOR_HSV2RGB)
                if not os.path.isdir(save_path+'/pcd_segs'): os.mkdir(save_path+'/pcd_segs')
                plt.imsave(save_path+'/pcd_segs/'+args.output_prefix+'_'+str(frame_number).zfill(4)+'.png',pcd_segs_rgb_img)
                
                    
                bev_img = cv2.resize(bev_img,(height,width))
                masked_bev_img = cv2.resize(masked_bev_img,(height,width))
                felz_segs_masked_bev_img = cv2.resize(felz_segs_masked_bev_img,(height,width))
                pcd_segs_img = cv2.resize(pcd_segs_img,(height,width))

                if args.pcd_segment:
                        if args.mask and args.felz_segment:
                                _frame = np.hstack((pcd_segs_rgb_img,masked_dist_img*0,pcd_segs_img))
                        elif args.mask:
                                _frame = np.hstack((pcd_segs_rgb_img,pcd_segs_img))
                        elif args.felz_segment:
                                _frame = np.hstack((pcd_segs_rgb_img,pcd_segs_img))
                        else:
                                _frame = pcd_segs_rgb_img
                                
                        frame = np.vstack((frame,_frame))

                if args.view and args.felz_segment:
                    _frame = np.hstack((bev_img,masked_bev_img,felz_segs_masked_bev_img))
                elif args.view:
                    _frame = np.hstack((bev_img,masked_bev_img))

                if args.view and frame is not None:
                    frame = np.vstack((frame,_frame))
                elif args.view:
                    frame = _frame
                    

                frame = cv2.resize(frame,None,fx=2,fy=2)
                cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
                cv2.putText(frame, str(frame_number), (15, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                if frame_number == 0:
                        size = (frame.shape[1],frame.shape[0])
                        
                        out = cv2.VideoWriter(save_path+'/'+args.output_prefix+".mp4", fourCC, 15.0, size, True)
                out.write(frame)
                
out.release()
#inlier_cloud = pcd.select_down_sample(_inliers)
# cmap = plt.get_cmap("tab20")
# colors = cmap(NMlabels / (NMmax_label if NMmax_label > 0 else 1))
# colors[NMlabels <0] = 0
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd])
# f, axs = plt.subplots(2,1,figsize=(15,15))
# legs0 = []
# legs1 = []
# for k in counts.keys():
#         if 'masked' in k:
#                 legs1.append(k)
#                 axs[1].plot(counts[k][5:])
#         else:
#                 legs0.append(k)
#                 axs[0].plot(counts[k][5:])
#         #plt.show()
# lgd = axs[0].legend(legs0,bbox_to_anchor=(1.1, 1.05))
# lgd = axs[1].legend(legs1,bbox_to_anchor=(1.1, 1.05))

#plt.savefig(save_path+"/"+args.output_prefix+".png",box_extra_artists=(lgd,), bbox_inches='tight')
d = dict(open3d_clusters_size,**counts)
scipy.io.savemat(save_path+'/'+args.output_prefix+".mat",d)
