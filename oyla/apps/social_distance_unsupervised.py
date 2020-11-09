##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Version:           2.0 2.08.2020
# Description                                                                                    #
# this is the unsupervised social distancing script -- it does motion subtraction using   cv2     #
# on color coded depth images. And then does 3D clustering on the masked point cloud             #
# the clusers are bounded in a box and centroid of clusters is used to calculate distance between #
# clusters and if distance of a cluster from any other is less than threshold its bbox is red else green#
# y,z and all that is needed for bev and like in depth_views_generation its about framing the veiw-- we should check.
##################################################################################################
import numpy as np
import struct
import sys
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
#from oyla.mvc.model_read_bin_data import DataRead
from oyla.utils import convert_matrix_image


import argparse
import json
import distutils.util
import open3d as o3d
import scipy.io
import glob
from oyla.unsupervised.utils import depth_to_xyz_view_img, perspective_projection
from oyla.apps.utils import read_input_config

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",type = str,required = True)
parser.add_argument("--segmentation_sigma",type=float, default = 1.5)
parser.add_argument("--segmentation_scale",type=float, default = 500)
parser.add_argument("--segmentation_min_size",type=int, default = 50)
parser.add_argument("--segmentation_eps",type=int, default = 100)
parser.add_argument("--x_max",type = float, default = None)
parser.add_argument("--x_min",type = float, default = None)
parser.add_argument("--y_max",type = float, default = None)
parser.add_argument("--y_min",type = float, default = None)
parser.add_argument("--z_min",type=int,default=None)
parser.add_argument("--z_max",type=int,default=None)
parser.add_argument("--distance_threshold",type = float, default = 200)
parser.add_argument("--no_motion_subtraction", action="store_true")
parser.add_argument("--ground_thresholded",action="store_true")
parser.add_argument("--output_file_prefix",default='social_distance_unsupervised',help='give output file name')
parser.add_argument("--number_frames",type = int, default = -1)

args = parser.parse_args()

if args.ground_thresholded: args.no_motion_subtraction = True
print(args.no_motion_subtraction,args.ground_thresholded)


save_path = args.input_dir+'/collateral/'
if not os.path.isdir(save_path): os.mkdir(save_path)
output_file_prefix = args.output_file_prefix
if not args.no_motion_subtraction: output_file_prefix += "_ms"
if args.ground_thresholded: output_file_prefix += "_gt"


segmentation_scale = args.segmentation_scale
segmentation_sigma = args.segmentation_sigma
segmentation_min_size = args.segmentation_min_size
segmentation_eps = args.segmentation_eps

pcl_files = sorted(glob.glob(args.input_dir+'/pcl*.xyz'))
rgb_files = sorted(glob.glob(args.input_dir+'/rgb*.png'))
depth_files = sorted(glob.glob(args.input_dir+'/depth*.mat'))
fourCC = cv2.VideoWriter_fourcc(*'MP4V')

dist_backSub = cv2.createBackgroundSubtractorMOG2()
rgb_backSub = cv2.createBackgroundSubtractorMOG2()
open3d_clusters_size = {}
frame_number = -1

x_max, x_min, y_max, y_min, z_max, z_min, range_max, range_min = read_input_config(args)


for pcl_file, rgb_file, depth_file in zip(pcl_files,rgb_files,depth_files):
    frame_number += 1
    pcd = o3d.io.read_point_cloud(pcl_file,format='xyz')
    print("\nNew frame",frame_number, pcl_file, rgb_file, depth_file)
    rgb_img = np.uint8(plt.imread(rgb_file)[:,:,:3]*255)
    if frame_number == 0:
        width = rgb_img.shape[0]
        height = rgb_img.shape[1]
        # this is needed for the bev projection, views from pcd

                 

    _tmp = scipy.io.loadmat(depth_file)
    dist = _tmp['dist']
    ampl = _tmp['ampl']
    #print(np.count_nonzero(dist),np.count_nonzero(ampl))
    dist_img = convert_matrix_image(dist,cmap= 'jet_r', clim_min=range_min, clim_max=range_max,
                                                   no_data_indices = np.where(dist==0))[:,:,:3]

    dist_fgMask = np.ones_like(dist)*255
    if not args.no_motion_subtraction:
        dist_fgMask = dist_backSub.apply(dist_img)        

    elif args.ground_thresholded:
        xyz = np.asarray(pcd.points)
        _tmp = np.where(dist >0)
        _ind = np.where(xyz[:,0]>np.median(xyz[:,0])*0.9)[0]
        dist_fgMask[_tmp[0][_ind],_tmp[1][_ind]] = 0

    masked_dist = np.zeros_like(dist)
    masked_dist[dist_fgMask==255] = dist[dist_fgMask==255]
    non_mask_indices = np.where(dist_fgMask != 255)
    mask_indices = np.where(dist_fgMask == 255)
    masked_dist_img = convert_matrix_image(masked_dist,cmap= 'jet_r', clim_min=range_min, clim_max=range_max,
                                               no_data_indices = non_mask_indices)[:,:,:3]
    rgb_fgMask = rgb_backSub.apply(rgb_img)
    masked_rgb_img = np.zeros_like(rgb_img)
    masked_rgb_img[rgb_fgMask==255] = rgb_img[rgb_fgMask==255]
    

    dist_flatten = dist.flatten()
    indices_dist_flatten = np.where(dist_flatten>0)[0]
    masked_indices_dist_flatten = np.ravel_multi_index(mask_indices,dist.shape)
    #print(np.setdiff1d(mask_indices_dist_flatten,indices_dist_flatten).shape,mask_indices_dist_flatten.shape,indices_dist_flatten.shape)
    #print(masked_dist_flatten[np.setdiff1d(mask_indices_dist_flatten,indices_dist_flatten)])
    _,_,_masked_indices = np.intersect1d(masked_indices_dist_flatten,indices_dist_flatten,return_indices=True)
    #print(_masked_indices.shape,np.unique(_masked_indices).shape,pcd,np.count_nonzero(dist_flatten))
    try:
        masked_pcd = pcd.select_by_index(_masked_indices)
    except AttributeError:
        masked_pcd = pcd.select_down_sample(_masked_indices)
    
    #pcd_segs = np.zeros_like(masked_dist,dtype = np.uint16)
    _mE = [0,0]
    max_label = 0
    labels = np.zeros_like(np.asarray(masked_pcd.points))
    if len(_masked_indices)>0:
        labels = np.array(masked_pcd.cluster_dbscan(eps=segmentation_eps, min_points=segmentation_min_size, print_progress=False))
        max_label = labels.max()
        print('Open3D Masked unique labels with counts',frame_number, np.unique(labels,return_counts=True))
        open3d_clusters_size[str(frame_number)] = np.unique(labels,return_counts=True)
        xyz = np.asarray(masked_pcd.points)
        _t = np.c_[xyz,labels]
        masked_dist_flat = masked_dist.flatten()
        masked_indices_dist_flatten = np.where(masked_dist.flat>0)[0]
        _mE = []
        _mI = []
        _mA = []
        
        for l in np.unique(labels):
            
            #print(l,labels==l,indices_nonzero_flat)
            masked_dist_flat[masked_indices_dist_flatten[labels==l]] = l+1
            _mE.extend(np.mean(_t[np.where(_t[:,3]==l),:3],axis=1))
            #_mI.extend(np.min(_t[np.where(_t[:,3]==l),:3],axis=1))
            #_mA.extend(np.max(_t[np.where(_t[:,3]==l),:3],axis=1))
        open3d_clusters_size[str(frame_number)] = np.vstack((open3d_clusters_size[str(frame_number)][0].transpose(),open3d_clusters_size[str(frame_number)][1].transpose(),np.asarray(_mE).transpose()))
        pcd_segs = masked_dist_flat.reshape(masked_dist.shape)
        
    #(x_max,y_max,z_max) = np.max(np.asarray(masked_pcd.points),0)
    #(x_min,y_min,z_min) = np.min(np.asarray(masked_pcd.points),0)
    #bev,VA,VB = depth_to_xyz_view_img(masked_pcd,view='yz',a_max = y_max, a_min = y_min, b_max = z_max, b_min = z_min, labels = labels+1,
    #                                   output_width = width, output_height = height)
    #print(bev.shape)
    #print(np.unique(bev,return_counts=True))
    bev = perspective_projection(masked_pcd, type_of_projection = 'yz', col_max = y_max, col_min = y_min, row_max = z_max, row_min = z_min, labels = labels+1,
                                 num_rows = height, num_cols = width)
    
    bev = np.rot90(bev)
    #print(np.unique(bev))
    #_t = bev.astype('float32')/np.max(bev+0.000001)
    #bev_img = cv2.cvtColor((_t*255).astype('uint8'),cv2.COLOR_GRAY2RGB)
    #no_data_indices = np.where(bev==0)
    #bev_img = convert_matrix_image(bev,cmap= 'jet_r', clim_min=0, clim_max=50, no_data_indices = no_data_indices)[:,:,:3]
    _bev = bev-np.min(bev)
    _bev /= np.max(_bev)
    bev_img = np.uint8(cm.get_cmap('tab20')(_bev)*255)[:,:,:3]
    

    try:
        centroid_coords = open3d_clusters_size[str(frame_number)][2:5,:]
        #min_coords = open3d_clusters_size[str(frame_number)][5:8,:]
        #max_coords = open3d_clusters_size[str(frame_number)][8:11,:]
        #print(coords.transpose())
    except KeyError:
        centroid_coords = np.zeros((3,1))
        #min_coords = np.zeros((3,1))
        #max_coords = np.zeros((3,1))
    
    for l in np.unique(pcd_segs).astype(int):
        if l == 0:
            continue
        if centroid_coords[0,l] <0:
            continue
        top_left = np.min(np.where(pcd_segs==l),axis=1)
        bottom_right = np.max(np.where(pcd_segs==l),axis=1)
        #print(l,top_left,bottom_right)
        
        _tmp = []
        for k in np.unique(pcd_segs).astype(int):
            if k == 0:
                continue
            if k !=l:
                _tmp.append(np.sqrt(np.sum((centroid_coords[:,l]-centroid_coords[:,k])**2)))
        
        #print(l,centroid_coords[:,l],np.where(np.asarray(_tmp)<=args.distance_threshold))
        if np.all(np.asarray(_tmp)>args.distance_threshold):
            cv2.rectangle(rgb_img, (top_left[1], top_left[0]), (bottom_right[1], bottom_right[0]), (0, 255, 0),1)
        if np.any(np.asarray(_tmp)<=args.distance_threshold):
            cv2.rectangle(rgb_img, (top_left[1], top_left[0]), (bottom_right[1], bottom_right[0]), (255, 0, 0),1)

        # top_left = np.min(np.where(bev==l),axis=1)
        # bottom_right = np.max(np.where(bev==l),axis=1)
        # #print(top_left, bottom_right)
        # if np.any(np.asarray(_tmp)>500):
        #     cv2.rectangle(bev_img, (top_left[1], top_left[0]), (bottom_right[1], bottom_right[0]), (0,0,255),10)
        # else:
        #     cv2.rectangle(bev_img, (top_left[1], top_left[0]), (bottom_right[1], bottom_right[0]), (0,255,0),10)



    pcd_segs -= np.min(pcd_segs)
    pcd_segs /= np.max(pcd_segs)
    pcd_segs_img = np.uint8(cm.get_cmap('tab20')(pcd_segs)*255)[:,:,:3]
    
    
    _frame = np.hstack((dist_img,masked_dist_img))
    _frame = np.hstack((_frame, pcd_segs_img))

    bev_img = cv2.resize(bev_img,(height,width))
    frame = np.hstack((rgb_img,bev_img))
    frame = np.hstack((frame, pcd_segs_img*0))
    
    frame = np.vstack((_frame,frame))
    #frame = np.hstack((dist_img,pcd_segs_img,rgb_img))
    #plt.imsave('tmp.png',frame)
    frame = cv2.resize(frame,None,fx=2,fy=2)
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(frame_number), (15, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    if frame_number == 0:
        size = (frame.shape[1],frame.shape[0])
        
        out = cv2.VideoWriter(save_path+'/'+output_file_prefix+".mp4", fourCC, 15.0, size, True)
    
    out.write(frame)
    
    if args.number_frames> 0 and frame_number > args.number_frames:
       break
                
out.release()
scipy.io.savemat(save_path+'/'+output_file_prefix+".mat",open3d_clusters_size)
