import sklearn.neighbors
import scipy.io
import numpy as np
from scipy.cluster.vq import whiten
import time
import scipy.spatial
dir_path='/Users/rsingh/Oyla Dropbox/Oyla/02 Technical/'+'10 Data/44x33/02_Jan_data_January_02_17_04_03///'
A = scipy.io.loadmat(dir_path+'/data_1_26.mat')['data']

depth = A[0][2][0][0][:,:,0]
ampl = A[0][2][0][0][:,:,1]

(ii,jj) =np.where(ampl<65000)
_ampl = ampl[ii,jj]
_depth = depth[ii,jj]

features = np.vstack((ii,jj,_ampl)).transpose()

features_w = whiten(features)

B = sklearn.neighbors.kneighbors_graph(features_w, 100,mode='distance',n_jobs=100)

def graph_median(i):
    return np.median(_depth[scipy.sparse.find(B[i,:])[1]])

# r = []
# start = time.time()
# for i in range(B.shape[0]):
#     r.append(graph_median(i))
# end = time.time()
# print("running throught for ",end-start)

from joblib import Parallel, delayed
import multiprocessing
my_list = range(B.shape[0])
start = time.time()
results = []
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores, verbose=0)(delayed(
    graph_median)(i)for i in my_list)
end = time.time()
print("running through joblib",end-start)
#print(r)
#print(results)
#assert r==results

# indices = scipy.sparse.find(B)

# from numba import njit, prange
# @njit(parallel=True)

# def jitp(L,indicesR,indicesC):
#     r = np.zeros(L)
#     for i in prange(L):
#         #r.append(np.median(index[1][np.where(index[0]==i)]))
#         #print(i)
#         r[i] = np.median(_depth[indicesC[np.where(indicesR==i)]])
#     return r


# start = time.time()
# rr = jitp(B.shape[0],indices[0],indices[1])
# end = time.time()
# print("numba",end-start)
# print(rr)
# assert r==rr

import open3d as o3d

pcd = o3d.geometry.PointCloud()
start = time.time()
pcd.points = o3d.utility.Vector3dVector(features_w)
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

def pcd_search(kk):
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[kk], 100)
    return idx    
rrr = np.zeros(B.shape[0])
for kk in range(B.shape[0]):
    idx = pcd_search(kk)
    rrr[kk] = np.median(_depth[idx])
end = time.time()
print('open3d',end-start)
#import matplotlib.pyplot as plt
#print(np.vstack((rrr,results)))
#assert (np.asarray(results)==rrr).all()

# start = time.time()
# tree = scipy.spatial.KDTree(features_w)
# pts = tree.query(features_w,k=100)
# idx = pts[1]
# print(idx.shape)
# end = time.time()
# print('scipy',end-start)

# results2 = []
# results2 = Parallel(n_jobs=num_cores, verbose=0)(delayed(
#     pcd_search)(i)for i in my_list)
# end = time.time()
# print('open3d joblib',end-start)
