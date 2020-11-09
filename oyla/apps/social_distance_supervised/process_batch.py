import queue
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from matplotlib import pyplot as plt
import yaml
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize

from detectron2.modeling import build_model
import torch
from detectron2.checkpoint import DetectionCheckpointer
import concurrent.futures
import pandas as pd
from oyla.unsupervised.utils import depth_to_xyz_view_img, depth_to_xyz_coords, perspective_projection
from oyla.utils import  some_common_utility, convert_matrix_image
from matplotlib import cm
from process_frame import ProcessFrame
import time
import queue
import pickle
import csv

from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict

from utils import euclidean_distance_with_thresholding

from PIL import Image
from filterpy.kalman import KalmanFilter

class CentroidTracker():
    def __init__(self, maxDisappeared=70, maxDistance=300, conf=.95): #140
        self.nextObjectID = 0
        self.centroids = OrderedDict()
        self.new_centroids = OrderedDict()
        self.boxes = OrderedDict()
        self.disappeared = OrderedDict()
        self.kalman = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.conf = conf

    def getCentroidKeys(self):
        return list(self.new_centroids.keys())

    # def getTotalNumPeople(self):
    #     return self.nextObjectID

    def register(self, centroid, bbox):
        print("registering objectID", self.nextObjectID)
        self.centroids[self.nextObjectID] = centroid
        self.new_centroids[self.nextObjectID] = centroid
        self.boxes[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0 

        ## Kalman filter
        # dim_x = 4 (2 dimensions - y and z, 2 variables - position and velocity)
        # dim_z = 2 measurements - y and z positions
        kf = KalmanFilter(dim_x=4, dim_z=2) 
        kf.x[:2] = centroid.reshape(2, 1)
        kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])   # State transition matrix
        kf.H = np.array([[1,0,0,0],[0,1,0,0]])  # Measurement function
        kf.R *= 1 #0.1                          # Measurement noise - state uncertainty
        kf.P *= 1000.                           # Covariance matrix
        kf.Q *= 0.1                             # Process uncertainty #Q_discrete_white_noise(2, dt=0.1, var=0.1)
        self.kalman[self.nextObjectID] = kf

        self.nextObjectID += 1

    def deregister(self, objectID):
        print("de-registering objectID", objectID)
        del self.centroids[objectID]
        del self.new_centroids[objectID]
        del self.boxes[objectID]
        del self.disappeared[objectID]
        del self.kalman[objectID]

    def update(self, inputCentroids, inputBoxes, scores):
        ### STILL NEED TO FIGURE OUT HOW TO UPDATE CENTROIDS NOT DETECTED IN CURRENT FRAME ###
        ### Right now, new_centroids not doing anything ###
        # print("nextObjectID", self.nextObjectID)
        print("STARTING centroids", self.centroids)
        # self.new_centroids = self.centroids.copy()
        for i in list(self.kalman.keys()):
            self.kalman[i].predict()
            self.new_centroids[i] = ((self.kalman[i]).x[:2]).flatten()
            # self.centroids[i] = ((self.kalman[i]).x[:2]).flatten()
        print("PREDICTED centroids", self.new_centroids)
        if len(inputCentroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
                self.kalman[objectID].update(self.centroids[objectID])
            # return np.array(list(self.centroids.values())), np.array(list(self.boxes.values()))
            return np.array(list(self.new_centroids.values())), np.array(list(self.boxes.values()))
        if len(self.new_centroids) == 0:
            for i in range(len(inputCentroids)):
                if scores[i] > self.conf:
                    self.register(inputCentroids[i], inputBoxes[i]) 
        else:
            objectIDs = list(self.new_centroids.keys())
            objectCentroids = np.array(list(self.new_centroids.values()))
            D = dist.cdist(inputCentroids, objectCentroids)
            row_ind, col_ind = linear_sum_assignment(D)
            print("objectIDs", objectIDs)
            print("inputCentroids\n", inputCentroids)
            print("objectCentroids\n", objectCentroids)           
            print("D", D)
            print("row_ind", row_ind)
            print("col_ind", col_ind)
            print("scores", scores)
            for i in range(len(row_ind)):
                # print("i ", i, "cost: ", D[i, col_ind[i]])
                if D[row_ind[i], col_ind[i]] < self.maxDistance:
                    # print("less than max dist")
                    self.centroids[objectIDs[col_ind[i]]] = inputCentroids[row_ind[i]]
                    self.new_centroids[objectIDs[col_ind[i]]] = inputCentroids[row_ind[i]]
                    self.kalman[objectIDs[col_ind[i]]].update(inputCentroids[row_ind[i]])
                    self.boxes[objectIDs[col_ind[i]]] = inputBoxes[row_ind[i]]
                    self.disappeared[objectIDs[col_ind[i]]] = 0
                else:
                    # print("not less than max dist")
                    if scores[row_ind[i]] > self.conf:
                        self.register(inputCentroids[row_ind[i]], inputBoxes[row_ind[i]])
            for row in range(len(inputCentroids)):
                if row not in row_ind:
                    if scores[row] > self.conf:
                        self.register(inputCentroids[row], inputBoxes[row])
            print("centroids: ", self.centroids)
            print("new_centroids: ", self.new_centroids)          
            for col in range(len(objectCentroids)):
                if col not in col_ind:
                    objectID = objectIDs[col]
                    # if self.disappeared[objectID] % 5 == 0:
                        # Set to last good measurement but set measurement noise variance to large number
                        # self.kalman[objectID].R = self.disappeared[objectID] * 5
                        # self.kalman[objectID].update(self.centroids[objectID])
                        # self.kalman[objectID].update(((self.kalman[objectID]).x[:2]).flatten())
                    self.disappeared[objectID] += 1
                    self.boxes[objectID] = torch.as_tensor(np.array([0., 0., 0., 0.]).astype("float32"))
                    if self.disappeared[objectID] > self.maxDisappeared:
                        # print("deregistered object ", objectID )
                        self.deregister(objectID)
        # print("FINAL centroids", centroids)
        return np.array(list(self.new_centroids.values())), torch.stack(list(self.boxes.values()))

#class ProcessBatch:
#    def __init__(self, output_dict, score = 0, flag_nms = None, flag_centroid_calculation = 'mean', distance_threshold = 200,
#                 flag_segmentation = False, flag_vis = False, flag_region = False, y_max = None, y_min = None, z_max = None, z_min = None,
#                 out_file_prefix = "/Users/divya/Desktop", scale_factor = 2, flag_white = False, flag_pcl = False, isometric = False, x_max = None, x_min = None):



class ProcessBatch:
    def __init__(self, output_dict, score = None, flag_nms = False, flag_centroid_calculation = 'mean', distance_threshold = 200, bev_shape = (360, 588),
                 flag_segmentation = False, flag_vis = False, flag_region = False, y_max = None, y_min = None, z_max = None, z_min = None, x_max = None, x_min = None,
                 out_file_prefix = "~/Desktop", scale_factor = 2, flag_white = False, flag_pcl = False, flag_all_distances = False,
                 flag_tracking = False, isometric = False,):
       
        cfg = get_cfg()

        if flag_segmentation:
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            # cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
            # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
        else:
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
        print(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score if score is not None else 0.9
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score if score is not None else 0.7

#        if flag_nms:
#            cfg.MODEL.RETINANET.NMS_THRESH_TEST = 1.0
#            cfg.TEST.DETECTIONS_PER_IMAGE = 1000

        if flag_nms is not None:
            cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.8
            cfg.TEST.DETECTIONS_PER_IMAGE = 100

        cfg.MODEL.DEVICE = 'cpu'
        self.model = build_model(cfg)
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS)
        self.model.eval()

        self.min_size_test = cfg.INPUT.MIN_SIZE_TEST
        self.max_size_test = cfg.INPUT.MAX_SIZE_TEST

        self.process_frame = ProcessFrame(flag_nms = flag_nms, flag_centroid_calculation = flag_centroid_calculation, distance_threshold = distance_threshold, flag_segmentation = flag_segmentation)
        self.output_dict = output_dict
        self.cfg = cfg
        self.flag_vis = flag_vis
        self.flag_segmentation = flag_segmentation
        self.flag_region = flag_region
        self.flag_white = flag_white
        self.flag_pcl = flag_pcl
        self.flag_all_distances = flag_all_distances
        self.flag_tracking = flag_tracking
        self.y_max = y_max
        self.y_min = y_min
        self.z_min = z_min
        self.z_max = z_max
        self.incident_counter = [0 for i in range(5)]
        self.people_counter = [0 for i in range(5)]
        self.sliding_counter_idx = 0
        self.log = out_file_prefix + "output.csv"
        self.scale_factor = scale_factor
        self.distance_threshold = distance_threshold
        self.tracker = CentroidTracker()
        self.total_people = OrderedDict()
        self.contact0 = OrderedDict()

        ### Get BEV voxel size ###
        self.bev_width = bev_shape[0]
        self.bev_height = bev_shape[1]
        VOXEL_SIZE_y = (self.y_max - self.y_min)/self.bev_width
        VOXEL_SIZE_z = (self.z_max - self.z_min)/self.bev_height
        self.VOXEL_SIZE =  max(VOXEL_SIZE_y, VOXEL_SIZE_z)
        self.y_offset= round((self.bev_width - (self.y_max-self.y_min)/self.VOXEL_SIZE) / 2) if VOXEL_SIZE_y < VOXEL_SIZE_z else 0
        self.z_offset = 0 if VOXEL_SIZE_y < VOXEL_SIZE_z else round((self.bev_height - (self.z_max-self.z_min)/self.VOXEL_SIZE) / 2) 

        with open(self.log, 'w') as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(["Frame Number", "Area of region", "Number people in region ", "Number incidents in region", "Score", "Total people", "Contact with person 0"])

        self.x_max = x_max
        self.x_min = x_min
        self.sliding_counter = [0 for i in range(5)]
        self.people_counter = [0 for i in range(5)]
        self.sliding_counter_idx = 0
        self.isometric = isometric
        theta = np.radians(45)
        c, s = np.cos(theta), np.sin(theta)
        R1 = np.array(((1,0,0),(0,c, s),  (0,-s, c)))
        theta = np.radians(35.26)
        c, s = np.cos(theta), np.sin(theta)
        R2 = np.array(((c, 0, -s),(0,1,0), (s, 0,c)))
        self.R = np.dot(R1,R2)


        
    def thread_funct(self, input_dict, stop):
        start_time=time.time()
        print("NEW THREAD")
        # if stop(): 
        #     print('exiting')
        # return

        a = time.perf_counter()
        with torch.no_grad():
            predictions=self.model(input_dict['image_list'])
        b = time.perf_counter()
        print("Inference time: ", b-a, "seconds")
        
        pickle_output_dict=self.output_dict['pickle_out']

        num=0
        file_number_list = input_dict['file_number_list']
        thread_list={}
        # thread_start_time={}
        # que = queue.Queue()


        for (prediction,pcd_array,image,depth,depth_img) in zip(predictions,input_dict['pcd_array_list'],input_dict['rgb_list'], input_dict['depth_list'],input_dict['depth_img_list']):


            frame_no = file_number_list[num]
            print("FRAME NUMBER ", frame_no)


            scale_factor = self.scale_factor
            width = image.shape[1] * scale_factor
            height = image.shape[0] * scale_factor

            thread_id, complete_output, centroids_xyz, mask_xyz = self.process_frame.working_on_each_frame(prediction, pcd_array , depth, lambda : thread_list[thread_id])
            frame = image
            # thread_id = threading.Thread(target=lambda q, arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,args9:
            #                              q.put(working_on_each_frame(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,args9)), 
            #                              args=(que,output,pcd_array,image,IJ,depth,num,input_dict['flag'],input_dict['flag_nms'],lambda : thread_list[thread_id]))
            # thread_list[thread_id]= False
            # thread_start_time[thread_id]=time.time()
            # thread_id.start()

            pickle_output_dict[str(file_number_list[num])+'_complete_output']=complete_output
            pickle_output_dict[str(file_number_list[num])+'_centroids_xyz']=centroids_xyz
            labels = np.ones_like(pcd_array[:,0])*-1
            
            ### Draw visualization output from detectron ###
            if self.flag_vis:
                v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.0)
                out = v.draw_instance_predictions(complete_output.to("cpu"))
                frame = out.get_image()[:, :, ::-1]

            frame = cv2.UMat(frame).get()
            frame = cv2.resize(frame, (width, height))

            ### Draw frame number ###
            cv2.rectangle(frame, (10, 30), (100,58), (255,255,255), -1)
            cv2.putText(frame, str(frame_no), (20, 55),cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,0,0))

            ### Get point cloud labels ###
            if self.flag_pcl:
                if self.flag_segmentation:
                    for i, person in enumerate(mask_xyz):
                        for pcd_idx in person:
                            labels[pcd_idx] = int(i)
                else:
                    label = 0
                    index = np.ravel_multi_index(np.where(depth>0),depth.shape)
                    for i in complete_output._fields['pred_boxes'].tensor:
                        _d = depth.copy()
                        _d[_d>0] = 1
                        _d[int(i[1]):int(i[3]),int(i[0]):int(i[2])] *= -1
                        ind = np.ravel_multi_index(np.where(_d==-1),_d.shape)
                        _,ind,_ = np.intersect1d(index,ind,return_indices = True)
                        labels[ind] = label
                        label += 1
                        del _d
                    
                    # for i in selected_output._fields['pred_boxes'].tensor:
                    #     _d = depth.copy()
                    #     _d[_d>0] = 1
                    #     _d[int(i[1]):int(i[3]),int(i[0]):int(i[2])] *= -1
                    #     ind = np.ravel_multi_index(np.where(_d==-1),_d.shape)
                    #     _,ind,_ = np.intersect1d(index,ind,return_indices = True)
                    #     labels[ind] = label
                    #     label += 1
                    #     del _d

            if self.flag_white:
                bev_img = np.ones((self.bev_height,self.bev_width,3), np.uint8)*255
                color = (0, 0, 0)
            else:
                bev_img = np.zeros((self.bev_height,self.bev_width,3), np.uint8)
                color = (255, 255, 255)

            ### Draw scale ###
            cv2.line(bev_img, (20, self.bev_height-40), (20+round(6*30.48/self.VOXEL_SIZE), self.bev_height-40), color, thickness=2)
            cv2.putText(bev_img, "6 ft", (25, self.bev_height-45),cv2.FONT_HERSHEY_SIMPLEX, 1 , color, thickness=2)

            # Divya some issue with heat map
            # ### Draw score heat map ###
            heat_x = self.bev_width - 75 #1110
            heat_y = 30
            heat_w = 40 #20
            heat_h = 150 #100
            heat = np.array([[i]*heat_w for i in range(heat_h)]) #np.array([i for i in range(heat_h)]*heat_w).reshape(heat_w,heat_h).transpose()
            heat = np.uint8(heat*255/heat_h)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_HOT)
            bev_img[heat_y:heat_y+heat_h,heat_x:heat_x+heat_w,:] = heat

            ### Calculate xy positions of centroids (and point clouds) on BEV image ###
            # print("CENTROIDS XYZ: ", centroids_xyz)
            centroids_yz = np.array(centroids_xyz)[:,1:] if len(centroids_xyz)>0 else []
            if self.flag_tracking:
                tracked_centroids, tracked_boxes = self.tracker.update(centroids_yz, complete_output._fields['pred_boxes'].tensor, complete_output._fields['scores'])
                centroid_keys = self.tracker.getCentroidKeys()
            else:
                tracked_centroids = centroids_yz
                tracked_boxes = complete_output._fields['pred_boxes'].tensor
                centroid_keys = [i for i in range(len(centroids_yz))]
            # print("TRACKED CENTROIDS: \n", tracked_centroids)
            # print("TRACKED BOXES: \n", tracked_boxes)

            complete_distances, selected_distances=euclidean_distance_with_thresholding(tracked_centroids, self.distance_threshold)
            # print("COMPLETE DISTANCES \n", complete_distances)
            # print("SELECTED_DISTANCES \n", selected_distances)

            selected_idx=[]
            for i in selected_distances.keys():
                if i[0] not in selected_idx:
                    selected_idx.append(i[0])
                if i[1] not in selected_idx:
                    selected_idx.append(i[1])

            ### Calculate xy positions of centroids (and point clouds) on BEV image ###

            centroid_bev = []
            for I, (y,z) in enumerate(tracked_centroids):
                # print("YZ COORDS: ", y, z)
                y = int((y - self.y_min) / self.VOXEL_SIZE + self.y_offset)
                z = int((z - self.z_min) / self.VOXEL_SIZE + self.z_offset)  
                centroid_bev.append((y, self.bev_height-z))
                # print("BEV cv2 COORDS: ", y, self.bev_height-z)

                #=======
                #            centroids = np.array(points_selected)
                #            centroid_xyz = []
                #            for I, centroid in enumerate(centroids):
                #                if np.any(centroid) is None:
                #                    centroid_xyz.append((None, None))
                #                    continue
                #                x,y,z = centroid[0:3]
                #                # print("Centroid "+str(I)+": ")
                #                # print("XYZ COORDS: ", x, y, z)
                #                if VOXEL_SIZE_y < VOXEL_SIZE_z:
                #                    y = int((y - self.y_min) / VOXEL_SIZE + offset)
                #                    z = int((z - self.z_min) / VOXEL_SIZE)
                #                else:
                #                    y = int((y - self.y_min) / VOXEL_SIZE)
                #                    z = int((z - self.z_min) / VOXEL_SIZE + offset)
                #                centroid_xyz.append((y, height-z))
                #                # print("BEV cv2 COORDS: ", y, height-z)
                #                # print("Backcalculate xyz coords")
                #                # y_ = (y-offset)*VOXEL_SIZE + self.y_min
                #                # z_ = (height-(height-z))*VOXEL_SIZE + self.z_min
                #                # print("XYZ COORDS backcalculated: ", x, y_, z_)
                #            print(len(centroids), len(centroid_xyz))
                #>>>>>>> Stashed changes
            if self.flag_pcl:
                labels=labels+1
                labels_color =  np.random.randint(0, 255, (len(centroids_xyz)+1, 3)) 
                labels_color = [(int(x[0]), int(x[1]), int(x[2])) for x in labels_color.tolist()]
                # labels_color = [(101, 153, 49), (40, 140, 212), (171, 70, 199), (109, 59, 160), (154, 6, 43), (209, 220, 11), (224, 104, 95)]
                for I, point in enumerate(pcd_array):
                    x,y,z = point[0:3]
                    # Threshold elevation at max 275
                    if x > 275:
                        continue
                    y = int((y - self.y_min) / self.VOXEL_SIZE + self.y_offset)
                    z = int((z - self.z_min) / self.VOXEL_SIZE + self.z_offset)
                    try:
                        test = bev_img[height-z, y]
                        if labels[I] == 0:     # Only label predicted classes
                            continue
                        cv2.circle(bev_img, (y, height-z), 2, labels_color[int(labels[I])], -1) 
                    except:
                        pass

            ### Draw region of interest - adjust these  ###
            if self.bev_width == 640:    # CDS 
                region_x = 190 #400 #40
                region_y = 160 #200 #40
                region_width = 200 #250
                region_height = 100 #250 #300
            else:               # DUS
                region_x = 385 #400 #40
                region_y = 320 #200 #40
                region_width = 400 #250
                region_height = 200 #250 #300

            # Print yz coordinations of region of interest box
            # y1 = (region_x-self.y_offset)*self.VOXEL_SIZE + self.y_min
            # z1 = (self.bev_height-(region_y-self.z_offset))*self.VOXEL_SIZE + self.z_min
            # y2 = (region_x+region_width-self.y_offset)*self.VOXEL_SIZE + self.y_min
            # z2 = (self.bev_height-(region_y+region_height-self.z_offset))*self.VOXEL_SIZE + self.z_min
            # print("region xyz coords", y1, z1, y2, z2)

            if self.flag_all_distances:
                idx_in_region = [i for i in range(len(centroid_bev))]
                selected_distances_in_region = complete_distances
                selected_idx_in_region = idx_in_region
            elif self.flag_region:
                idx_in_region = [i for i in range(len(centroid_bev)) if (0<=centroid_bev[i][0]-region_x<=region_width) and (0<=centroid_bev[i][1]-region_y<=region_height)]
                selected_distances_in_region = dict(filter(lambda elem: (elem[0][0] in idx_in_region and elem[0][1] in idx_in_region), selected_distances.items()))
                selected_idx_in_region = np.unique(np.array(list(selected_distances_in_region.keys())).flatten())
            else:
                idx_in_region = [i for i in range(len(centroid_bev))]
                selected_distances_in_region = selected_distances
                selected_idx_in_region = selected_idx

            ### Count incidents and calculate scores (sliding window average of number of incidents)###
            num_incidents = len(selected_distances_in_region)
            num_people = len(idx_in_region)
            if frame_no == 0:
                self.incident_counter = [num_incidents for i in range(5)]
                self.people_counter = [num_people for i in range(5)]
            else:
                self.incident_counter[self.sliding_counter_idx] = num_incidents
                self.people_counter[self.sliding_counter_idx] = num_people
            self.sliding_counter_idx = (self.sliding_counter_idx + 1)%5
            # score = sum(self.incident_counter)/(sum(self.people_counter)+0.00000000001)
            score = min(sum(self.incident_counter), 24)/24     # 24 arbitrary number to normalize score for heatmap

            ### Draw box around region of interest ###
            if self.flag_region:
                # Switch below to change region box color according to score
                # c = np.uint8(np.array([1-score])*255)
                # c = cv2.applyColorMap(c, cv2.COLORMAP_AUTUMN)
                # c = tuple(c[0][0])
                # region_color = (int(c[0]), int(c[1]), int(c[2]))
                # cv2.rectangle(bev_img, (region_x, region_y), (region_x+region_width, region_y+region_height), color=region_color, thickness=2)
                
                cv2.rectangle(bev_img, (region_x, region_y), (region_x+region_width, region_y+region_height), color, thickness=2)

            ### Draw statistics in upper left corner ###
            cv2.putText(bev_img, "People: "+str(num_people), (15, 55),cv2.FONT_HERSHEY_SIMPLEX, 1 , color, thickness=2)

            cv2.putText(bev_img, "Incidents: "+str(num_incidents), (15, 90),cv2.FONT_HERSHEY_SIMPLEX, 1 , color, thickness=2) 
            cv2.arrowedLine(bev_img, (heat_x-25, heat_y+heat_h-round(score*heat_h)), (heat_x-1, heat_y+heat_h-round(score*heat_h)), color, 2, tipLength=0.25)   

            ### Draw centroids and distances on BEV frame and color coded bounding boxes on RGB images ###
            ### Red for violators in region red, green for non-violators in region, white/black for outside region
            for i, bbox in enumerate(tracked_boxes):#complete_output._fields['pred_boxes'].tensor):
                id_color = (255, 255, 255)
                #=======
                #            cv2.putText(bev_img, "Incidents: "+str(num_violations), (15, 95),cv2.FONT_HERSHEY_SIMPLEX, 1 , color, thickness=2)
                #            cv2.rectangle(bev_img, (10, 2), (100,20), (255,255,255), -1)
                #            cv2.putText(bev_img, str(frame_no), (15, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                #            #cv2.arrowedLine(bev_img, (heat_x-25, heat_y+heat_h-round(score*heat_h)), (heat_x-1, heat_y+heat_h-round(score*heat_h)), color, 2, tipLength=0.25)
                #
                #            ### Draw centroids and distances on BEV frame and color coded bounding boxes on RGB images ###
                #            for i, bbox in enumerate(complete_output._fields['pred_boxes'].tensor):
                #                # Raghav
                #                if np.any(points_selected[i]) is None:
                #                    continue
                #                ## Draw centroids - color violators in region red, non-violators in region green, outside region white/black
                #>>>>>>> Stashed changes
                if i in selected_idx_in_region:
                    bbox_color = (0, 0, 255)
                elif i in idx_in_region:
                    bbox_color = (0, 255, 0)
                else:
                    bbox_color = color
                    id_color = (255, 255, 255) if color==(0, 0, 0) else id_color
                cv2.circle(bev_img, centroid_bev[i], 9, bbox_color, -1) 
                if bbox != [0, 0, 0, 0]:
                    bbox = bbox * scale_factor
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]),int(bbox[3])), bbox_color, 1)
                if self.flag_tracking:
                    if i not in selected_idx_in_region:
                        cv2.putText(bev_img, str(centroid_keys[i]), (centroid_bev[i][0]-5, centroid_bev[i][1]+5), cv2.FONT_HERSHEY_SIMPLEX, .5, id_color, 2)
                    if i in idx_in_region:
                        self.total_people[i] = 1

            for i, bbox in enumerate(tracked_boxes):
                ## Draw distances on BEV (between violators only)
                for j in range(i+1, len(tracked_boxes)):
                    if i not in selected_idx_in_region or j not in selected_idx_in_region:
                        #=======
                        #                    if self.flag_white:
                        #                        bbox_color = (0, 0, 0)
                        #                    else:
                        #                        bbox_color = (255, 255, 255)
                        #                # Draw centroid on BEV
                        #                cv2.circle(bev_img, centroid_xyz[i], 6, bbox_color, -1)
                        #                # Draw color coded bounding boxes on RGB
                        #                bbox = bbox * scale_factor
                        #                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]),int(bbox[3])), bbox_color, 1)
                        #                # Label point number for debugging
                        #                # cv2.putText(bev_img, str(i), centroid_xyz[i], cv2.FONT_HERSHEY_SIMPLEX, .25, (255, 255, 255), 1)
                        #
                        #                for j in range(i+1, len(centroid_xyz)):
                        #                    ## Draw distances on BEV (between violators only)
                        #                    if i not in selected_idx_in_region or j not in selected_idx_in_region or np.any(points_selected[j]) is None :
                        #>>>>>>> Stashed changes
                        continue
                    try: 
                        distance = selected_distances_in_region[(i,j)]
                        text_x = np.clip((centroid_bev[i][0]+centroid_bev[j][0])//2 - 2, 0, self.bev_width-1)
                        text_y = np.clip((centroid_bev[i][1]+centroid_bev[j][1])//2 + 5, 0, self.bev_height-1)
                        text = '{0:.1f}'.format(distance/30.48) #cm to ft
                        cv2.line(bev_img, centroid_bev[i], centroid_bev[j], (0, 0, 255), thickness=2)                        
                        if self.flag_tracking:
                            cv2.putText(bev_img, str(centroid_keys[i]), (centroid_bev[i][0]-5, centroid_bev[i][1]+5), cv2.FONT_HERSHEY_SIMPLEX, .5, id_color, 2)
                            cv2.putText(bev_img, str(centroid_keys[j]), (centroid_bev[j][0]-5, centroid_bev[j][1]+5), cv2.FONT_HERSHEY_SIMPLEX, .5, id_color, 2)
                            if centroid_keys[i] == 0:
                                self.contact0[centroid_keys[j]] = 1
                            elif centroid_keys[j] == 0:
                                self.contact0[centroid_keys[i]] = 1
                        cv2.putText(bev_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, .6, color, thickness=2)
                    except:
                        pass

            # if self.flag_tracking:
            #     cv2.putText(bev_img, "Total People: "+str(len(self.total_people)), (15, 125),cv2.FONT_HERSHEY_SIMPLEX, 1 , color, thickness=2) 
            #     cv2.putText(bev_img, "Contact w/0: "+str(len(self.contact0)), (15, 160),cv2.FONT_HERSHEY_SIMPLEX, 1 , color, thickness=2) 

            with open(self.log, 'a') as fp:
                csv_writer = csv.writer(fp)
                region_area = round(region_height * region_width * self.VOXEL_SIZE * self.VOXEL_SIZE / 10000)
                csv_writer.writerow([str(frame_no), str(region_area), num_people, num_incidents, score*24/5, len(self.total_people), len(self.contact0)])            
            
            ### Stack RGB and BEV images
            frame = np.hstack((frame,bev_img))
            print('fs',frame.shape)
            if self.isometric:
                iso = perspective_projection(pcd_array,type_of_projection='xy', col_max = self.x_max, col_min = self.x_min, row_max = self.y_max, row_min = self.y_min,
                                             isometric_R = self.R, voxel_row_size = 4, voxel_col_size = 4)
                iso = np.rot90(iso)
                #iso = np.flipud(iso)
                # _iso = iso-np.min(iso)
                # _iso /= (np.max(_iso)+0.000001)
                no_data_indices = np.where(iso==0)
                iso_img = convert_matrix_image(iso,cmap= 'jet', clim_min=self.z_min, clim_max=self.z_max, no_data_indices = no_data_indices)[:,:,:3]
                # iso_img = np.uint8(cm.get_cmap('viridis')(iso)*255)[:,:,:3]
                # iso_img = cv2.UMat(iso_img).get()
                iso_img = cv2.resize(iso_img,(width, height))
                depth_img = cv2.resize(depth_img,(width, height))
                _frame = np.hstack((depth_img,iso_img))
                frame = np.vstack((frame,_frame))
                
            self.output_dict['video_out'].write(frame)
            num += 1

        #pickle.dump(pickle_output_dict, self.output_dict['pickle_out'])
        
        # for thread_id in thread_list.keys():
        #     if stop():
        #         thread_list[thread_id]=True
        #     thread_id.join()

        # _frame=[]

        # selected_op_list={}
        # frame_list={}
        # complete_output_list={}
        # starting_file_number=input_dict['file_number_list'][0]
        # pickle_out=output_dict['pickle_out']
        # while not que.empty():
        #     num,selected_output,frame,complete_output=que.get()
        #     selected_op_list[num]=selected_output
        #     frame_list[num]=frame
        #     complete_output_list[num]=complete_output

        # for i in range(len(selected_op_list)):
        #     selected_output=selected_op_list[i]
        #     frame=frame_list[i]
        #     complete_output=complete_output_list[i]
        #     pickle_output_dict[str(starting_file_number+num)+'complete_output']=complete_output_list[i]
        #     pickle_output_dict[str(starting_file_number+num)+'selected_output']=selected_op_list[i]
        #     frame = cv2.UMat(frame).get()
        #     for i in complete_output._fields['pred_boxes'].tensor:
        #         cv2.rectangle(frame, (int(i[0]), int(i[1])), (int(i[2]),int(i[3])), (0,255,0), 1)
        #     for i in selected_output._fields['pred_boxes'].tensor:
        #         #             print(i)
        #         cv2.rectangle(frame, (int(i[0]), int(i[1])), (int(i[2]),int(i[3])), (0,0,255), 1)
        #         output_dict['video_out'].write(frame)
        #         pickle.dump(pickle_output_dict, pickle_out)
        print('DONE------------->',(time.time()-start_time))
