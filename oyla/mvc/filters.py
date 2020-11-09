##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Creation:          20.04.2019                                                                  #
# Version:           1.0                                                                         #
# Revision history:  Initial version                                                             #
# Description                                                                                    #
#         filter functions using cv2 library, spatial_edge needs more info                       #
##################################################################################################

import cv2
import numpy as np
import pywt
from skimage.filters import median
from skimage.morphology import disk,square
from skimage.util import img_as_ubyte
from skimage.restoration import wiener
import scipy.cluster.vq
import open3d as o3d
import scipy.sparse
import sklearn.neighbors
#import sys
#sys.path.append('/Users/rsingh/Packages/l0_gradient_minimization_test')
#from l0_gradient_minimization import l0_gradient_minimization_2d

def clip_img(img):
    return np.clip(img, 0, 1)
def filters_in_use(img, guide, filter_params, phase_min, phase_max, ambiguity_distance):
    
    assert img.dtype == 'float32', 'img to be filtered has to be float 32'
    assert np.max(img)<=1.0, 'img to be filteredhas to be in range[0..1] max'
    assert np.min(img)>=0.0, 'img to be filtered has to be in range[0..1] min'
    assert guide.dtype == 'float32', 'guide has to be float 32'
    #guide = guide/np.max(guide)
    assert np.max(guide)<=1.0, 'guide to  filtered has to be in range[0..1] max'
    assert np.min(guide)>=0.0, 'guide to be filtered has to be in range[0..1] min'

    img = filter_spatial_wavelet_GraceVet(img,filter_params).astype('float32')
    img = clip_img(img)

    img = filter_spatial_pcl_median(guide,img,filter_params)
    img = filter_spatial_pcl_weighted_average(guide,img,filter_params)
    
    #img = filter_morphological(img, filter_params)
    img = filter_spatial_median(img, filter_params)
    guide =  filter_spatial_median(guide, filter_params)
    img = filter_spatial_masked_median(img,filter_params)
    img = filter_spatial_gaussian(img, filter_params)
    #img = filter_spatial_wiener(img,filter_params)
    img = filter_spatial_bilateral(img, filter_params)
    guide = filter_spatial_bilateral(guide, filter_params)
    
    img = filter_spatial_guided(guide, img, filter_params)
    img = filter_spatial_jointbilateral(guide, img, filter_params)
    #img = filter_l0_gradient_minimize(img,filter_params)
    img = filter_spatial_edge(img, filter_params, phase_min, phase_max, ambiguity_distance)
    img = filter_spatial_pcl_outliers(guide,img,filter_params)
        
    return img, guide

def filter_recipie(img,guide,filter_params,phase_min,phase_max,ambiguity_distance):
    filter_params = {}
    filter_params['median_filter'] = 0
    filter_params['median_filter_size'] = 3
    filter_params['median_filter_iterations'] = 1
    img = filter_spatial_median(img,filter_params)
    filter_params = {}
    filter_params['edge_detection'] = 0
    filter_params['edge_detection_thresholds'] = 40
    img = filter_spatial_edge(img, filter_params, phase_min, phase_max, ambiguity_distance)
    fiter_params = {}
    filter_params['guided_filter'] = 1
    filter_params['guided_filter_iterations'] = 1
    filter_params['guided_filter_size'] = 3
    filter_params['guided_filter_std_range'] = 0.01
    img = filter_spatial_guided(guide, img, filter_params)
    filter_params = {}
    filter_params['edge_detection'] = 0
    filter_params['edge_detection_thresholds'] = 10
    img = filter_spatial_edge(img, filter_params, phase_min, phase_max, ambiguity_distance)
    return img


def filter_spatial_median(data, filter_params):
    filter_size = 3
    filter_iterations = 3
    if 'median_filter' in filter_params and int(filter_params['median_filter']):
        print("Doing median filtering")
        if 'median_filter_size' in filter_params:
            filter_size = int(filter_params['median_filter_size'])
        if 'median_filter_iterations' in filter_params:
            filter_iterations = int(filter_params['median_filter_iterations'])
        #data = data.astype('float32')
        for _ in range(filter_iterations):
            _data = cv2.medianBlur(data,filter_size)
            print(np.where(np.logical_and(_data==0, data!=0))[0].shape,np.where(np.logical_and(_data!=0, data==0))[0].shape, np.mean((_data-data)**2))
            data = _data
    return data

def filter_temporal_median(median_array):
    #print(median_array.shape)
    return np.median(median_array[:, :, :, 0], axis=0), \
        np.median(median_array[:, :, :, 1], axis=0)

def filter_spatial_masked_median(data, filter_params):
    #has an issue in that its flattening in some wierd spatial way
    filter_size = 1
    filter_iterations = 1

    if 'masked_median_filter' in filter_params and int(filter_params['masked_median_filter']):
        print("Doing masked median filtering")
        mask = data.copy()
        mask[mask>0] = 1.0
        
        #_data = img_as_ubyte(data)
        _data = data.astype('float32')
        if 'masked_median_filter_size' in filter_params:
            filter_size = int(filter_params['masked_median_filter_size'])
        if 'masked_median_filter_iterations' in filter_params:
            filter_iterations = int(filter_params['masked_median_filter_iterations'])
        
        for _ in range(filter_iterations):
            _data = median(_data,square(filter_size),behavior='rank',mask=mask)
        data = _data
        #data = _data.astype('float32')/255.0
        
    return data

def filter_spatial_pcl_outliers(guide,data,filter_params):
    ind = []
    if 'pcl_outliers_filter' in filter_params and int(filter_params['pcl_outliers_filter']):
        print("Doing pcl outliers_filtering")
        [ii,jj] = np.where(data>0)

        _guide = guide[ii,jj]
        _data = data[ii,jj]
        features = scipy.cluster.vq.whiten(np.vstack((ii,jj,_data)).transpose())
        #features = (np.vstack((ii.astype('float32')/np.max(ii),jj.astype('float32')/np.max(jj),_data)).transpose())
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(features)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
        out = np.setdiff1d(np.arange(ii.shape[0]),ind)
        #print(len(out))
        _data[out] = 0

        data[ii,jj] = _data
        #ind = np.where(data==-1)
        #data[ind] = 0
        # outlier_cloud = pcd.select_down_sample(ind, invert=True)
        # inlier_cloud = pcd.select_down_sample(ind)
        # inlier_cloud.paint_uniform_color([0.5,0.5,0.5])
        # o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud])
        # print(inlier_cloud,outlier_cloud)
        del pcd
    return data

def filter_spatial_pcl_median(guide,data,filter_params):

    if 'pcl_median_filter' in filter_params and int(filter_params['pcl_median_filter']):
        print("Doing pcl median filtering")
        [ii,jj] = np.where(data>0)
        # kk = np.ravel_multi_index((ii,jj),data.shape)
        # kkk = np.hstack((kk,kk+1,kk-1,kk+data.shape[1],kk-data.shape[1]))
        # kkk = kkk[np.where(kkk>0)]
        # kkk = kkk[np.where(kkk<data.shape[0]*data.shape[1])]
        # kkk = np.unique(kkk)
        # [ii,jj] = np.unravel_index(kkk,data.shape)
        _guide = guide[ii,jj]
        _data = data[ii,jj]
        
        features = (np.vstack((ii.astype('float32')/np.max(ii),jj.astype('float32')/np.max(jj),_guide)).transpose())
        #features = scipy.cluster.vq.whiten(np.vstack((ii.astype('float32'),jj.astype('float32'),_data)).transpose())
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(features)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        __data = np.zeros_like(_data)
        #__data = _data
        for kk in range(_data.shape[0]):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[kk],9)
            #[k, idx, _] = pcd_tree.search_hybrid_vector_3d(pcd.points[kk],0.02,49)
            #print(idx,_data[idx])
            __data[kk] = np.median(_data[idx])
        print(np.where(np.logical_and(__data==0, data[ii,jj]!=0))[0].shape,np.where(np.logical_and(__data!=0, data[ii,jj]==0))[0].shape,np.mean((__data-data[ii,jj])**2))
        data[ii,jj] = __data
        
        del pcd, pcd_tree
    return data

def filter_spatial_pcl_weighted_average(guide,data,filter_params):

    if 'pcl_weighted_average_filter' in filter_params and int(filter_params['pcl_weighted_average_filter']):
        print("Doing pcl weigthed average filtering")
        import time
        start = time.time()
        [ii,jj] = np.where(data>0)
        _guide = guide[ii,jj]
        _data = data[ii,jj]
        print(np.std(data),np.std(_guide))
        features = (np.vstack((ii.astype('float32')/np.max(ii),jj.astype('float32')/np.max(jj),_guide)).transpose())
        
        #print(time.time()-start)
        A = sklearn.neighbors.kneighbors_graph(features, 1,mode='distance',n_jobs=100)
        #print(A.shape[0],time.time()-start)
        A = (-100*A.power(2)).expm1()
        #print(time.time()-start)
        A.data += 1
        #print(time.time()-start)
        A += scipy.sparse.eye(A.shape[0])
        #print(time.time()-start)
        D = np.squeeze(np.asarray(np.sum(A,axis=1)))
        D = 1/D
        #print(time.time()-start)
        iD= scipy.sparse.coo_matrix(A.shape)
        iD.setdiag(D)
        #print(time.time()-start)
        __data = iD.dot(A.dot(_data))
        print(np.where(np.logical_and(__data==0, data[ii,jj]!=0))[0].shape,np.where(np.logical_and(__data!=0, data[ii,jj]==0))[0].shape,np.mean((__data-data[ii,jj])**2))
        #print(time.time()-start)
        data[ii,jj] = __data
        
    return data


def filter_spatial_wiener(data, filter_params):
    filter_size = 3
    balance = 1.1
    _psf = 'gaussian'
    if 'wiener_filter' in filter_params and int(filter_params['wiener_filter']):
        print("Doing wiener filter")
        if 'wiener_filter_size' in filter_params:
            filter_size = int(filter_params['wiener_filter_size'])
        if 'wiener_filter_balance' in filter_params:
            balance = float(filter_params['wiener_filter_balance'])
        if 'wiener_filter_psf' in filter_params:
            _psf = filter_params['wiener_filter_psf']
        if _psf == 'gaussian':
            psf = cv2.getGaussianKernel(filter_size,-1)
            psf = np.matmul(psf,psf.transpose())
        else:
            psf = np.ones((filter_size,filter_size))/np.power(filter_size,2)
        data = wiener(data,psf,balance)
    return data

def filter_morphological(data, filter_params):
    filter_size = 3
    if 'morphological_filter' in filter_params and int(filter_params['morphological_filter']):
        print("Doing morphological_filter")
        if 'morphological_filter_size' in filter_params:
            filter_size = int(filter_params['morphological_filter_size'])
        kernel = disk(filter_size)
        data = cv2.morphologyEx(data,cv2.MORPH_CLOSE,kernel)
        data = cv2.morphologyEx(data,cv2.MORPH_OPEN,kernel)
    return data

def filter_l0_gradient_minimize(data,filter_params):
    lmd = 0.002
    beta_max = 1.0e5
    beta_rate = 2.0
    enhancement = 1.5
    if 'l0_gradient_filter' in filter_params and int(filter_params['l0_gradient_filter']):
        print("Doing l0 gradient minimization")
        data_base = l0_gradient_minimization_2d(data, lmd, beta_max, beta_rate)
        data_diff = data - data_base
        data_enhance = clip_img(data_base + data_diff*enhancement)
        data = data_base
    return data

def filter_spatial_gaussian(data, filter_params):
    filter_size = 3
    filter_std_space = 0.1
    filter_iterations = 1
    if 'gauss_filter' in filter_params and int(filter_params['gauss_filter']):
        print("Doing Gaussian filtering")
        if 'gauss_filter_iterations' in filter_params:
            filter_iterations = int(filter_params['gauss_filter_iterations'])
        if 'gauss_filter_size' in filter_params:
            filter_size = int(filter_params['gauss_filter_size'])
        if 'gauss_filter_std_space' in filter_params:
            filter_std_space = float(filter_params['gauss_filter_std_space'])
        #data = data.astype('float32')
        #filter = np.asarray([[1,2,1],[2,4,2],[1,2,1]])/16.0 #consistent with filter in espros client
    
        for _ in range(filter_iterations):
            data = cv2.GaussianBlur(data,(filter_size,filter_size),sigmaX = filter_std_space)
            #data = cv2.filter2D(data,-1,filter)
    return data

def filter_spatial_bilateral(data,filter_params):
    from skimage.restoration import denoise_bilateral
    filter_size=-1
    filter_std_space=7.0
    filter_std_val=0.05
    filter_iterations = 1

    if 'bilateral_filter' in filter_params and int(filter_params['bilateral_filter']):
        print("Doing Bilateral filtering")
        if 'bilateral_filter_size' in filter_params:
            filter_size = int(filter_params['bilateral_filter_size'])
            if filter_size == -1:
                filter_size = None
        if 'bilateral_filter_iterations' in filter_params:
            filter_iterations = int(filter_params['bilateral_filter_iterations'])
        if 'bilateral_filter_std_space' in filter_params:
            filter_std_space = float(filter_params['bilateral_filter_std_space'])
        if 'bilateral_filter_std_range' in filter_params:
            filter_std_val = float(filter_params['bilateral_filter_std_range'])
            if filter_std_val == -1:
                filter_std_val = None
        print(np.std(data), filter_std_space, filter_std_val, filter_size)
        for _ in range(filter_iterations):
            #data = cv2.bilateralFilter(data,filter_size,sigmaSpace = filter_std_space,sigmaColor = filter_std_val)
            _data = denoise_bilateral(data,win_size = filter_size, sigma_color = filter_std_val, sigma_spatial = filter_std_space)
            print(np.where(np.logical_and(_data==0, data!=0))[0].shape,np.where(np.logical_and(_data!=0, data==0))[0].shape)
            
            data = _data
    #print(_m)
    return data

def filter_spatial_jointbilateral(joint,data,filter_params):
    filter_size=-1
    filter_std_space=7.0
    filter_std_val=0.05
    filter_iterations = 1
    

    if 'jointbilateral_filter' in filter_params and int(filter_params['jointbilateral_filter']):
        print("Doing JointBilateral filtering")
        if 'jointbilateral_filter_size' in filter_params:
            filter_size = int(filter_params['jointbilateral_filter_size'])
        if 'jointbilateral_filter_iterations' in filter_params:
            filter_iterations = int(filter_params['jointbilateral_filter_iterations'])
        if 'jointbilateral_filter_std_space' in filter_params:
            filter_std_space = float(filter_params['jointbilateral_filter_std_space'])
        if 'jointbilateral_filter_std_range' in filter_params:
            filter_std_val = float(filter_params['jointbilateral_filter_std_range'])
        #print(filter_size,filter_std_space,filter_std_val)
        
        
        for _ in range(filter_iterations):
            data = cv2.ximgproc.jointBilateralFilter(joint=joint,
                                                     src=data,
                                                     d = filter_size,
                                                     sigmaColor=filter_std_val,
                                                     sigmaSpace=filter_std_space)
            
    return data

def filter_spatial_guided(guide,data,filter_params):
    filter_size=9
    filter_std_val=0.05
    filter_iterations = 1
    
    if 'guided_filter' in filter_params and int(filter_params['guided_filter']):
        print("Doing Guided filtering")
        
        if 'guided_filter_size' in filter_params:
            filter_size = int(filter_params['guided_filter_size'])
        if 'guided_filter_iterations' in filter_params:
            filter_iterations = int(filter_params['guided_filter_iterations'])
        if 'guided_filter_std_range' in filter_params:
            filter_std_val = float(filter_params['guided_filter_std_range'])
        for _ in range(filter_iterations):
            data = cv2.ximgproc.guidedFilter(guide,
                                             src=data,
                                             radius=filter_size,
                                             eps=filter_std_val)
            #cv2.bilateralFilter(data,filter_size,sigmaSpace = filter_std_space,sigmaColor = filter_std_val)

    return data

# def filter_spatial_edge_(dist, range_min, ambiguity_distance_cm=1250, edge_detection_thresholds_mm = 100, mod_freq=12000000):
#     # There are bunch of calculations to be done based on the edge detection threshold
#     # the mm value (default) is .xml config file which is set as the MaxEdgeAmpDiff
#     # the phase value is calculated from the mm value (arg1) by:
    
#     # phaseEdgeThreshold = (double)distanceLabeler.getPhase(arg1){
#     # maxDistanceCM = set.getSpeedOfLightDiv2() / set.mod.getModulationFrequency() * 100;
#     # MAX_PHASE / getMaxDistanceCM() * arg1
#     # }
#     dist = dist.astype('float32')
#     MAX_PHASE = 30000.0
#     #this is conversion from edge_detection_threshold (in cm) to phase
#     phaseEdgeThreshold = edge_detection_thresholds_mm/10*MAX_PHASE/ambiguity_distance_cm
    
#     phaseEdgeThreshold = phaseEdgeThreshold/10.0
#     phaseEdgeThreshold += 0.5
    
#     # set.setMaxEdgeAmpDiff(arg1); //set mm
#     # set.setMaxEdgeAmpDiff((int)phaseEdgeThreshold, true); //set phase

#     #The above should lead us to getMaxEdgeAmpDiff (in phase)
#     maxEdgeAmpDiff= int(phaseEdgeThreshold)
#     #then we have to get minDistance
#     minPhase =  range_min*100*MAX_PHASE/ambiguity_distance_cm

#     maxEdgeAmpDiff = maxEdgeAmpDiff  * mod_freq/24000000
#     dX = dist.copy()*0
#     dY = dist.copy()*0
#     dXY = dist.copy()*0
#     edgeXdetect = 0
#     edgeYdetect = 0
#     for i in range(dist.shape[0]):
#         for j in range(dist.shape[1]-1):
#             diffY = np.abs(dist[i,j]- dist[i,j+1])
#             dY[i,j] = diffY

#             if(maxEdgeAmpDiff < diffY):
#                 edgeYdetect += 1
#             else:
#                 edgeYdetect = 0
                    
#             if maxEdgeAmpDiff < diffY and edgeYdetect!=1:
#                 dist[i,j] = minPhase
                    
                
            
#     for j in range(dist.shape[1]):
#         for i in range(dist.shape[0]-1):
#             diffX = np.abs(dist[i,j] - dist[i+1,j])
#             dX[i,j] = diffX
            
#             if(maxEdgeAmpDiff < diffX):
#                 edgeXdetect += 1
#             else:
#                 edgeXdetect=0;
                
#             if maxEdgeAmpDiff < diffX and edgeXdetect!=1:
#                 dist[i,j] = minPhase
#     return dist



def filter_spatial_edge(dist, filter_params, phase_min, phase_max, ambiguity_distance_cm=1250):
    ind = []
    if 'edge_detection' in filter_params and int(filter_params['edge_detection']):
        print("Doing edge detection")
        if 'edge_detection_thresholds' in filter_params:
            edge_detection_thresholds_cm = float(filter_params['edge_detection_thresholds'])
        
        dist = dist.astype('float32')
        MAX_PHASE = 30000.0
        #this is conversion from edge_detection_threshold (in cm) to phase
        phaseEdgeThreshold = edge_detection_thresholds_cm*MAX_PHASE/ambiguity_distance_cm
        #print ("Ambiguity distance", ambiguity_distance_cm)
        #print ("Edge Threshold", edge_detection_thresholds_cm)
        #print ("phaseEdgeThreshold", phaseEdgeThreshold)

        #phaseEdgeThreshold = phaseEdgeThreshold/10.0
        #phaseEdgeThreshold += 0.5
        #The above should lead us to getMaxEdgeAmpDiff (in phase)
        #maxEdgeAmpDiff= int(phaseEdgeThreshold)
        maxEdgeAmpDiff = phaseEdgeThreshold/phase_max
        #then we have to get minDistance
        minPhase =  phase_min/phase_max#(range_min)*MAX_PHASE/ambiguity_distance_cm 
        # range_min is passed directly from PCL window in cm and needs to be converted to phase
        
        #maxEdgeAmpDiff = maxEdgeAmpDiff  * mod_freq/24000000
        #print(maxEdgeAmpDiff,minPhase,ambiguity_distance_cm,mod_freq)
        

        # this seems to be a wierd filter but it basically ensures that singletons or first of a series are marked
        _filter = np.asarray([0, 2 , 1])

        dY = np.abs(dist[:,1:] - dist[:,:-1])>maxEdgeAmpDiff
        filtered = np.apply_along_axis(lambda m: np.convolve(m, _filter, mode='same'), axis=1, arr=dY)
        filtered = np.c_[filtered,np.zeros(dist.shape[0])]
        dist[filtered>2] = minPhase-1.0/phase_max
        #print(np.count_nonzero(filtered>2))
        dX = np.abs(dist[1:,:] - dist[:-1,:])>maxEdgeAmpDiff
        filtered = np.apply_along_axis(lambda m: np.convolve(m, _filter, mode='same'), axis=0, arr=dX)
        filtered = np.r_[filtered,np.   zeros((1,dist.shape[1]))]
        dist[filtered>2] = minPhase-1.0/phase_max
        #ind = np.where(filtered>2)
        #print(np.count_nonzero(filtered>2),edge_detection_thresholds_cm,phaseEdgeThreshold,maxEdgeAmpDiff)
    #print(np.mean(dist),np.std(dist))
    return dist
def bayes_shrink(c,sigma):
    v = np.power(c,2)
    v = np.sum(v)/np.prod(c.shape)
    sigmax = np.sqrt(np.max((v-sigma*sigma,0)))
    if sigmax:
        T = sigma*sigma/sigmax
    else:
        T = 0
    rc = np.sign(c)*np.max((np.abs(c)-T,np.zeros(c.shape)),axis=0)
    return rc

def filter_spatial_wavelet_GraceVet(dist,filter_params):
    if 'wavelet_GraceVet_filter' in filter_params and int(filter_params['wavelet_GraceVet_filter']):
        print("Doing wavelet GraceVet filtering")
        coeff = pywt.wavedec2(dist,'db5',level=3)
        sigma = np.median(np.abs(coeff[2][2]))/0.6745

        recon_coeff = []
        for i in range(4):
            if i ==0 :
                recon_coeff.append(bayes_shrink(coeff[i],sigma))
            else:
                _recon_coeff = []
                for j in range(3):
                    _recon_coeff.append(bayes_shrink(coeff[i][j],sigma))
                recon_coeff.append(_recon_coeff)
        #for i in range(1,4):
        #    recon_coeff[i] = tuple(recon_coeff[i])
        dist = pywt.waverec2(recon_coeff,'db5')
    return dist
