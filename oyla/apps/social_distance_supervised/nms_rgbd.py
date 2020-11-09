import numpy as np

from centroid_calculations import center_xyz_bfs
from iou_calc import get_iou
def nms_rgbd(boxes,scores,iou_threshold,depth_threshold,pcd_array,IJ,depth):
    b=np.asarray(boxes)
    keep=[i for i in range(b.shape[0])]
    scores=np.asarray(scores)
    for i in range(b.shape[0]):
        for j in range(i+1,b.shape[0]):
            if i not in keep or j not in keep:
                continue
            u=get_iou(b[i],b[j])
            #print('box',b[i],b[j])
#             print(u,scores[i],i,scores[j],j)
            if u>iou_threshold:
                center_a1=int((b[i][0]+b[i][2])/2)
                center_b1=int((b[i][1]+b[i][3])/2)
                center_a2=int((b[j][0]+b[j][2])/2)
                center_b2=int((b[j][1]+b[j][3])/2)
#                 print('centers1',center_a1,center_b1)
#                 print('centers2',center_a2,center_b2)
                c = np.asarray([center_a1,center_b1])
                c = c[np.newaxis,:]
                d_i=center_xyz_bfs(pcd_array,c,IJ,depth)
                c = np.asarray([center_a2,center_b2])
                c = c[np.newaxis,:]
                d_j=center_xyz_bfs(pcd_array,c,IJ,depth)
                #print('iou',u,d_i,d_j,scores[i],scores[j])
                #print('xyz',np.sum((d_i-d_j)**2)**0.5)
                if np.sum((d_i-d_j)**2)**0.5<depth_threshold:

                    d_i_mean=np.average(depth[int(b[i][0]):int(b[i][0]+b[i][2]),int(b[i][1]):int(b[i][1]+b[i][3])])
                    d_j_mean=np.average(depth[int(b[j][0]):int(b[j][0]+b[j][2]),int(b[j][1]):int(b[j][1]+b[j][3])])
                    print(i,j,d_i_mean,d_j_mean)
                    score_i=scores[i]+1/(np.log(d_i_mean)+0.00001)
                    score_j=scores[j]+1/(np.log(d_j_mean)+0.00001)
                    if score_i<score_j:
                        keep.remove(i)
                    else:
                        keep.remove(j)
                    #print('remove',len(keep))
    return keep
