# visited=np.zeros(dist.shape)
import numpy as np
import queue
from utils import ij_to_xyz_lookup


def bfs(a,b,IJ,dist):
    s1=dist.shape[0]
    s2=dist.shape[1]
#     print('shape',s1,s2)
    queue=[]
    if a>=s1 or b>=s2:
        return -1,-1,-1,-1
    if dist[a,b]!=0:
        pos=ij_to_xyz_lookup(a,b,IJ,dist.shape)
        return dist[a,b],a,b,int(pos)
    visited=np.zeros(dist.shape)
    queue.append([a,b])
    visited[a,b]=1
    while(len(queue)!=0):
        a1,b1=queue.pop(0)
#         print('a1',a1,'b1',b1)
        visited[a,b]=1
        if dist[a1,b1]!=0:
            pos=ij_to_xyz_lookup(a1,b1,IJ,dist.shape)
            return dist[a1,b1],a1,b1,int(pos)
        if a1+1<s1:
            if visited[a1+1,b1]==0:
                queue.append([a1+1,b1])
                visited[a1+1,b1]=1
            if b1+1<s2 and visited[a1+1,b1+1]==0:
                queue.append([a1+1,b1+1])
                visited[a1+1,b1+1]=1
            if b1+1<s2 and visited[a1,b1+1]==0:
                queue.append([a1,b1+1])
                visited[a1,b1+1]=1
            if b1-1>=0:
                if visited[a1+1,b1-1]==0:
                    queue.append([a1+1,b1-1])
                    visited[a1+1,b1-1]=1
                if visited[a1,b1-1]==0:
                    queue.append([a1,b1-1])
                    visited[a1,b1-1]=1
        if a1-1>=0:
            if visited[a1-1,b1]==0:
                queue.append([a1-1,b1])
                visited[a1-1,b1]=1
            if b1+1<s2 and visited[a1-1,b1+1]==0:
                queue.append([a1-1,b1+1])
                visited[a1-1,b1+1]=1
            if b1-1>=0 and visited[a1-1,b1-1]==0:
                queue.append([a1-1,b1-1])
                visited[a-1,b1-1]=1
#     print(visited[60,:])
    return -1,-1,-1,-1
