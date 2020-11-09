
import concurrent.futures

from utils import euclidean_distance_with_thresholding, ij_to_xyz_lookup
from centroid_calculations import *
import numpy as np
import time
from nms_rgbd import nms_rgbd
from soft_nms_pytorch import soft_nms_pytorch
from soft_nms import py_cpu_softnms
import detectron2


class ProcessFrame():
    def __init__(self,flag_nms = None, flag_centroid_calculation = 'mean', required_class = [0, 3], distance_threshold = 200, flag_segmentation = False):
        self.flag_nms = flag_nms
        self.flag_centroid_calculation = flag_centroid_calculation
        self.required_class = required_class
        self.distance_threshold = distance_threshold
        self.flag_segmentation = flag_segmentation
        
    def working_on_each_frame(self, prediction,pcd_array,depth, stop, num = -1):

        # if stop(): 
        #     print('exiting')
        #     return
        
        IJ=np.ravel_multi_index(np.where(depth!=0),depth.shape)
        mask_xyz = None

        if self.flag_segmentation:
            pred_classes = prediction['instances'].pred_classes.to("cpu").numpy()
            pred_masks = prediction['instances'].pred_masks.to("cpu")

            people = [i for i in range(len(pred_classes)) if pred_classes[i] in self.required_class] #np.argwhere(np.isin(pred_classes, self.required_class))
            # num_people = len(people)
            people_masks = [pred_masks[i] for i in range(len(pred_masks)) if i in people]

            unique_masks = []
            unique_people = []
            for i in range(len(people)):
                dup = 0
                for j in range(len(unique_masks)):
                    if dup:
                        continue
                    overlap = np.sum((people_masks[i] * (people_masks[i]==unique_masks[j])).numpy())
                    area_1 = np.sum(people_masks[i].numpy())
                    area_2 = np.sum(unique_masks[j].numpy())
                    if (overlap/min(area_1, area_2) > 0.8):
                        print("DUPLICATE")
                        dup = 1
                        # swap out for smaller mask
                        if np.sum(area_1 < area_2): 
                            unique_masks[j] = people_masks[i]
                            unique_people[j] = people[j]
                if dup == 0:
                    unique_masks.append(people_masks[i])
                    unique_people.append(people[i])
            # unique_masks = people_masks
            num_people = len(unique_people)
            mask_ij = [np.nonzero(unique_masks[i]) for i in range(num_people)]
            # mask_xyz = [ij_to_xyz_lookup(mask_ij[i][:,0],mask_ij[i][:,1],IJ,depth.shape) for i in range(num_people)]
            # hack to fix out of bounds error when set y_max limit
            mask_xyz = []
            for i in range(num_people):
                coord_list = ij_to_xyz_lookup(mask_ij[i][:,0],mask_ij[i][:,1],IJ,depth.shape)
                valid_coord_list = []
                for coord in coord_list:
                    try:
                        xyz = pcd_array[coord]
                        valid_coord_list.append(coord)
                    except:
                        pass
                if valid_coord_list != []:
                    mask_xyz.append(valid_coord_list)
                else:
                    del unique_people[i]

            if self.flag_centroid_calculation == 'median':
                points_selected = [np.median(pcd_array[mask_xyz[i]],axis=0) for i in range(len(mask_xyz))]
            else:
                points_selected = [np.mean(pcd_array[mask_xyz[i]],axis=0) for i in range(len(mask_xyz))]

            selected_boxes_from_prediction = unique_people

        else:
            print(prediction['instances'].get('scores').shape)
            if self.flag_nms is None:
                selected_boxes_from_prediction=detectron2.layers.nms(prediction['instances'].get('pred_boxes').tensor,prediction['instances'].get('scores'),0.2)#,10,IJ,depth)
            else:
                #selected_boxes_from_prediction=nms_rgbd(prediction['instances'].get('pred_boxes').tensor,prediction['instances'].get('scores'),0.5,25,pcd_array, IJ,depth)
                #soft_nms_pytorch(dets = prediction['instances'].get('pred_boxes').tensor,box_scores = prediction['instances'].get('scores'))
                selected_boxes_from_prediction = py_cpu_softnms(dets = np.asarray(prediction['instances'].get('pred_boxes').tensor), sc = np.asarray(prediction['instances'].get('scores')),
                                                                depth = depth, method = self.flag_nms)
            print('test nms',prediction['instances'].get('scores').shape,len(selected_boxes_from_prediction))
            selected_boxes_from_prediction=[int(i) for i in selected_boxes_from_prediction if prediction['instances'][int(i)]._fields['pred_classes'] in self.required_class]

            points_selected=[]

            #print(self.flag_centroid_calculation)

            if self.flag_centroid_calculation == 'centre_with_bfs':
                centers = prediction['instances'][selected_boxes_from_prediction]._fields['pred_boxes'].get_centers()
                centers = np.asarray(centers).astype(int)
                points_selected = center_xyz_bfs(pcd_array,centers,IJ,depth)

            else:
                for box in prediction['instances'][selected_boxes_from_prediction]._fields['pred_boxes'].tensor:
                    point = center_xyz_coordinate(pcd_array,IJ,depth,box, method = self.flag_centroid_calculation)
                    # if self.flag_centroid_calculation== 'mean':
                    #     #print(box)
                    #     point = mean_xyz_coordinate(pcd_array,IJ,depth,box)
                    # elif self.flag_centroid_calculation== 'mean_with_dbscan':
                    #     point = mean_xyz_coordinate_with_DBSCAN(pcd_array,IJ,depth,box)
                    # elif self.flag_centroid_calculation== 'median':
                    #     point = median_xyz_coordinate(pcd_array,IJ,depth,box)
                    #if point is not None:
                    points_selected.append(point)

        # distances, thresh_distances=euclidean_distance_with_thresholding(points_selected, self.distance_threshold)
        #=======
        #
        #        # if self.flag_region:
        #        #     points_selected = [points_selected[i] for i in range(len(points_selected)) if (points_selected[i]-region_)]
        #        #                 in_region = [i for i in range(len(centroid_xyz)) if (centroid_xyz[i][0]-region_x<=region_width) and (centroid_xyz[i][1]-region_y<=region_height)]
        #
        #
        #        distances, thresh_distances=euclidean_distance_with_thresholding(points_selected, self.distance_threshold)
        #>>>>>>> Stashed changes
        
        # people_closer_than_threshhold=[]
        # for i in thresh_distances.keys():
        #     if i[0] not in people_closer_than_threshhold:
        #         people_closer_than_threshhold.append(i[0])
        #     if i[1] not in people_closer_than_threshhold:
        #         people_closer_than_threshhold.append(i[1])

        # return num, prediction["instances"][selected_boxes_from_prediction][people_closer_than_threshhold],prediction["instances"][selected_boxes_from_prediction], thresh_distances, distances, people_closer_than_threshhold, points_selected, mask_xyz

        return num,prediction["instances"][selected_boxes_from_prediction], points_selected, mask_xyz

