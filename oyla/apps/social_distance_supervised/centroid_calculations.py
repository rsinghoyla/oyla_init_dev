import numpy as np
from utils import ij_to_xyz_lookup
from depth_completion_by_bfs import bfs
from sklearn.cluster import DBSCAN

import sys
sys.path.append("../")
from  oyla.utils import _transformation3_

def center_xyz_bfs(pcd_array,centers,IJ,depth):
    centeroid_vals=[]
    #print(centers)
    for i,j in centers:
        #visited=np.zeros(depth.shape)
#         print(i,j,np.asarray(IJ).shape,np.asarray(depth).shape)
#         print((visited==0).all())
        dis,a,b,pos=bfs(j,i,IJ,depth)
#         print(a,b,pos)
        if pos==-1:
            print('Position for image object not found')
            continue
        centeroid_vals.append(pos)
    return pcd_array[centeroid_vals]

def center_xyz_coordinate(pcd_array,IJ,depth,box, method = 'mean'):
        box=np.asarray(box.to("cpu")).astype(int)
        coordinates=[]

        x_list=[]
        y_list=[]
        # for x in range(int(box[0]),int(box[2])):
        #         for y in range(int(box[1]),int(box[3])):
                        
        #                 x_list.append(x)
        #                 y_list.append(y)
        x = range(int(box[0]),int(box[2]))
        y = range(int(box[1]),int(box[3]))
        
        z = depth[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
        x, y = np.meshgrid(x, y)
        
        x_list = x.reshape(-1)
        y_list = y.reshape(-1)
        z_list = z.reshape(-1)

        x_list = x_list[np.where(z_list>0)]
        y_list = y_list[np.where(z_list>0)]
        z_list = z_list[np.where(z_list>0)]
        #print(np.max(z_list))
        #coordinates = ij_to_xyz_lookup(y_list,x_list,IJ,depth.shape)
        
        #print('aaa',np.mean(dcoordinatesx),np.mean(dcoordinatesy),np.mean(dcoordinatesz))
        
        if len(z_list)>0:
            dcoordinatesx,dcoordinatesy,dcoordinatesz = _transformation3_(y_list, x_list, z_list, width = depth.shape[0], height = depth.shape[1], fov_angle = 33, fov_angle_o=44)
            # coordinates = []
            # print('???')
            # pos = 0
            # x=int((box[0]+box[2])/2)
            # y=int((box[0]+box[2])/2)
            # dis,a,b,pos = bfs(y,x,IJ,depth)
            # coordinates.append(pos)
            if method == 'mean':
                return np.mean(dcoordinatesx),np.mean(dcoordinatesy),np.mean(dcoordinatesz)#np.mean(pcd_array[coordinates],axis=0)
            elif method == 'median':
                return np.median(dcoordinatesx),np.median(dcoordinatesy),np.median(dcoordinatesz)#np.mean(pcd_array[coordinates],axis=0)
            elif method == 'mean_with_dbscan':
                db=DBSCAN(eps=100,min_samples=int(len(z_list)/4))
                
                
                db=db.fit(np.asarray([dcoordinatesx,dcoordinatesy,dcoordinatesz]).transpose())
                labels = db.labels_
                ulabels, counts = np.unique(labels,return_counts = True)
                print('dbscan clusters',ulabels, counts)
                largest_cluster = ulabels[np.where(ulabels>-1)][np.argmax(counts[np.where(ulabels>-1)])]
                return np.mean(dcoordinatesx[np.where(labels==largest_cluster)]), np.mean(dcoordinatesy[np.where(labels==largest_cluster)]), np.mean(dcoordinatesz[np.where(labels==largest_cluster)])
        else:
            return None, None, None 
def median_xyz_coordinate(pcd_array,IJ,depth,box):
        box=np.asarray(box.to("cpu")).astype(int)
        coordinates=[]

        x_list=[]
        y_list=[]
        # for x in range(int(box[0]),int(box[2])):
        #         for y in range(int(box[1]),int(box[3])):
                        
        #                 x_list.append(x)
        #                 y_list.append(y)
        x = range(int(box[0]),int(box[2]))
        y = range(int(box[1]),int(box[3]))
        x, y = np.meshgrid(x, y)
        x_list = x.reshape(-1)
        y_list = y.reshape(-1)

        coordinates = ij_to_xyz_lookup(y_list,x_list,IJ,depth.shape)

        if len(coordinates)==0:
            coordinates = []
            
            pos = 0
            x=int((box[0]+box[2])/2)
            y=int((box[0]+box[2])/2)
            dis,a,b,pos = bfs(y,x,IJ,depth)
            coordinates.append(pos)
        return np.median(pcd_array[coordinates],axis=0)

def mean_xyz_coordinate_with_DBSCAN(pcd_array,IJ,depth,box):
    db=DBSCAN(eps=100,min_samples=6)
    box=np.asarray(box).astype(int)
    coordinates=[]
    #     print(box)
    x_list=[]
    y_list=[]
    # for x in range(int(box[0]),int(box[2])):
    #     for y in range(int(box[1]),int(box[3])):
    #         x_list.append(x)
    #         y_list.append(y)
    x = range(int(box[0]),int(box[2]))
    y = range(int(box[1]),int(box[3]))
    x, y = np.meshgrid(x, y)
    x_list = x.reshape(-1)
    y_list = y.reshape(-1)
    coordinates = ij_to_xyz_lookup(y_list,x_list,IJ,depth.shape)

    if len(coordinates)==0:
        coordinates = []
        x=int((box[0]+box[2])/2)
        y=int((box[0]+box[2])/2)
        dis,a,b,pos= bfs(y,x,IJ,depth)
        coordinates.append(pos)
    
    centerx,centery=(int((box[0]+box[2])/2),int((box[1]+box[3])/2))
    dis,a,b,pos= bfs(centery,centerx,IJ,depth)
    xyz=pcd_array[pos]
    
    db=db.fit(pcd_array[coordinates])
    labels = db.labels_
    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    unique_labels = set(labels)
    
    for k in unique_labels:
        class_member_mask = (labels == k)
        if xyz in pcd_array[coordinates][class_member_mask & core_samples_mask]:
            if k==-1:
                print('IN NOISE')
            return np.mean(pcd_array[coordinates][class_member_mask & core_samples_mask],axis=0)
    print('NOT FOUND')
    return np.mean(pcd_array[coordinates],axis=0)
