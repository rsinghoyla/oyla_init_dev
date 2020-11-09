##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Version:           2.0 20.09.2020
# Description                                                                                    #
# Script to generate views from point cloud data -- point cloud data is projected (orthographic) to three planes, front,
# side, top and views are generated from that by pixelizing the 2d projections
# videos are made from the sequence of projections
# one problem is in pixelization -- how to best fit data into a frame, and related how best to quantize it in space.
# There is a default view and size of frame based on FOV and range of camera but that can be improved.
# Here also we also allow for x_max, x_min, y_max, y_min as command params; I am unclear whether they are exactly
# same as the params in read_raw_calibrate* -- there they server to threshold, here they serve to frame the projection
# z_min is also allowed if you want to frame in z direction --- note no thresholding is happening
# data outside the limits will be quantized to the limits itself
##################################################################################################
import numpy as np
import struct
import sys
import os
import cv2
import matplotlib.pyplot as plt
#from oyla.mvc.model_read_bin_data import DataRead
from oyla.utils import read_csv_parameters, some_common_utility, convert_matrix_image
from oyla.apps.utils import read_input_config
#from oyla.mvc.utils import CAMERA_VERSION, FOV
from oyla.unsupervised.utils import depth_to_xyz_view_img2
import argparse
import json
import distutils.util
import open3d as o3d
import scipy.io
from oyla.unsupervised.utils import *
import glob


# note that some of the UI parameters are here. missing range_m* assuming that that comes from parameters.csv
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",type = str,required = True)

parser.add_argument("--x_max",type = float, default = None)
parser.add_argument("--x_min",type = float, default = None)
parser.add_argument("--y_max",type = float, default = None)
parser.add_argument("--y_min",type = float, default = None)

parser.add_argument("--z_min",type=int,default=None)
parser.add_argument("--z_max",type=int,default=None)

parser.add_argument("--number_frames",type = int, default = -1)
parser.add_argument("--view",type=str, default='xy',help='xy, yz, xz')
parser.add_argument("--depth",type=lambda x:bool(distutils.util.strtobool(x)),default = False)
parser.add_argument("--rgb",type=lambda x:bool(distutils.util.strtobool(x)),default = True)
parser.add_argument("--isometric",type=lambda x:bool(distutils.util.strtobool(x)),default = False)
parser.add_argument("--factor",type=float, default = 0.9)
parser.add_argument("--output_file_prefix", type = str, default = "depth_views_video")


args = parser.parse_args()
output_file_prefix = args.output_file_prefix
if args.isometric:
        output_file_prefix += '_iso'
output_file_prefix +='_'+args.view

x_max, x_min, y_max, y_min, z_max, z_min, range_max, range_min = read_input_config(args)



    



pcl_files = sorted(glob.glob(args.input_dir+'/pcl*.xyz'))
rgb_files = sorted(glob.glob(args.input_dir+'/rgb*.png'))
depth_files = sorted(glob.glob(args.input_dir+'/depth*.png'))
total_num_of_images = len(depth_files)

fourCC = cv2.VideoWriter_fourcc('M','J','P', 'G'); # Important to notice cv2.cv.CV_FOURCC
    
        


save_path = args.input_dir+'/collateral'
if not os.path.isdir(save_path): os.mkdir(save_path)

pcd = o3d.geometry.PointCloud()
frame_number = -1
for pcl_file, rgb_file, depth_file in zip(pcl_files,rgb_files,depth_files):
        frame_number += 1
        print("\nNew frame",frame_number, pcl_file, rgb_file, depth_file)
        pcd = o3d.io.read_point_cloud(pcl_file,format='xyz')
        rgb_img = np.uint8(plt.imread(rgb_file)[:,:,:3]*255)[:,:,::-1]
        dist_img = np.uint8(plt.imread(depth_file)[:,:,:3]*255)
        #print(rgb_img.shape)
        height = rgb_img.shape[1]
        width = rgb_img.shape[0]
        
        if True: #imaging_type =='Dist_Ampl':

                    
                # for bev we will do y,z

                #z_max = range_max
                #z_min = range_min
                # y_max = args.y_max
                # y_min = args.y_min
                # x_max = args.x_max
                # x_min = args.x_min
                
                R = None
                if args.isometric:
                        theta = np.radians(45)
                        c, s = np.cos(theta), np.sin(theta)
                        R1 = np.array(((1,0,0),(0,c, s),  (0,-s, c)))
                        theta = np.radians(35.26)
                        c, s = np.cos(theta), np.sin(theta)
                        R2 = np.array(((c, 0, -s),(0,1,0), (s, 0,c)))
                        R = np.dot(R1,R2)
                        
                        #R = R1
                        #[x_max,y_max,z_max] = np.dot(R,np.asarray([x_max, y_max,z_max]))
                        #[x_min,y_min,z_min] = np.dot(R,np.asarray([x_min, y_min,z_min]))
                        #print(x_max,y_max,z_max, x_min, y_min, z_min)
                if args.view == 'xz':
                        bev = perspective_projection(pcd, type_of_projection = 'xz', col_max = x_max, col_min = x_min, row_max = z_max, row_min = z_min, isometric_R = R,
                                                     voxel_row_size = 2, voxel_col_size = 4)
                        clim_min, clim_max = y_min, y_max
                elif args.view == 'yz':
                        bev = perspective_projection(pcd, type_of_projection = 'yz', col_max = y_max, col_min = y_min, row_max = z_max, row_min = z_min, isometric_R = R,
                                                     voxel_row_size = 2, voxel_col_size = 4)
                        clim_min, clim_max = x_min, x_max
                elif args.view == 'xy':
                        bev = perspective_projection(pcd, type_of_projection = 'xy', col_max = x_max, col_min = x_min, row_max = y_max, row_min = y_min, isometric_R = R,
                                                     voxel_row_size = 4, voxel_col_size = 4)
                        clim_min, clim_max = z_min, z_max
                else:
                    bev = np.zeros((width,height))
                    args.view = False
                
                bev = np.rot90(bev)
                
                no_data_indices = np.where(bev==0)
                print(np.max(bev),np.min(bev[np.where(bev!=0)]))
                bev_img = convert_matrix_image(bev,cmap= 'jet', clim_min=clim_min, clim_max=clim_max, no_data_indices = no_data_indices)[:,:,:3]
                #                                    saturation_indices = indices['amplitude_saturated'],
                #                                    no_data_indices = indices['amplitude_low'],
                #                                    outside_range_indices = outside_range_indices)[:,:,:3]
                
                #print('xyz',bev_img.shape)  
                #plt.imshow(np.rot90(bev))
                #plt.pause(0.000001)
               
                if args.view:
                        _width, _height = bev_img.shape[:2]
                else:
                        _height = height
                        _width = width
                bev_img = cv2.resize(bev_img,(_height,_width))
                rgb_img = cv2.resize(rgb_img,(_height,_width))
                dist_img = cv2.resize(dist_img,(_height,_width))
                frame = None
                if args.depth:
                        frame = dist_img
                if args.rgb:
                        _frame = rgb_img
                        
                if args.depth and args.rgb:
                        frame = np.vstack((_frame,frame))
                elif args.rgb:
                        frame = _frame
                        
                
                if args.view and frame is not None:
                        frame = np.vstack((frame,bev_img))
                elif args.view:
                    frame = bev_img
                    
                #print('xyz',bev_img.shape,frame.shape)  
                #frame = cv2.resize(frame,None,fx=2,fy=2)
                if frame_number == 0:
                        size = (frame.shape[1],frame.shape[0])
                        
                        out = cv2.VideoWriter(save_path+'/'+output_file_prefix+".avi", fourCC, 15.0, size, True)
                out.write(frame)
                if args.number_frames> 0 and frame_number > args.number_frames:
                        break
                #plt.imsave('bev.png',frame)
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
