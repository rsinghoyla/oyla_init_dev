import numpy as np

def ij_to_xyz_lookup(y,x,IJ,dim):
        # if flag==0:
        #         try:
        #                 tempij=np.asarray(IJ)
        #                 temp1=np.where(tempij[0]==y)[0]
        #                 temp2=np.where(tempij[1][np.where(tempij[0]==y)[0]]==x)
        #                 pos=temp1[temp2]
        #                 pos=int(pos)
        #         except:
        #                 return -1
        # else:
        #         #try:
        #         tempij=np.asarray(IJ)
        #         temp1=[np.where(tempij[0]==i)[0] for i in y]
        #         temp2=[np.where(tempij[1][np.where(tempij[0]==i)[0]]==j) for (i,j) in zip(y,x)]
        #         pos=[i[j] for (i,j) in zip(temp1,temp2)]
        #         pos=[int(i) for i in pos if len(i)!=0]
        #         #except:
        #         #        print(y,x)
        #         #
        

        yx = np.ravel_multi_index((y,x),dim)
        
        pos = np.intersect1d(IJ,yx,return_indices=True)[1]
        return pos

def euclidean_distance_with_thresholding(points_selected, distance_threshold =200):
    distances={}
    thresh_distances={}
    points_selected=np.asarray(points_selected)
    for i in range(points_selected.shape[0]):
        for j in range(i+1,points_selected.shape[0]):

            if np.any(points_selected[i]) is None or np.any(points_selected[j]) is None:
                    distances[i,j] = None
                    continue
            
            # diff_sq=(points_selected[i]-points_selected[j])**2
            # dist=diff_sq[1]+diff_sq[2] #yz view only
            # dist=dist**0.5

            diff = points_selected[i]-points_selected[j]
            dist = np.sqrt(np.sum(diff**2))

            distances[(i,j)]=dist
            if dist <= distance_threshold:
                thresh_distances[(i,j)]=dist
    return distances, thresh_distances




