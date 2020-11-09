##################################################################################################
#                                                                                                #
# Author:            Oyla                                                                        #
# Creation:          20.04.2019                                                                  #
# Version:           1.0                                                                         #
# Revision history:  Initial version                                                             #
# Description                                                                                    #
#         utilties function like reading parameters from csv file, plotting,                     #
#                   animating frames to make movie                                               #
##################################################################################################


#import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import matplotlib.animation as animation
#import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy.io
from matplotlib import cm
from oyla.mvc.filters import filters_in_use, filter_recipie
from numba import jit
# (server.sendCommand("getModulationFrequencies")[::2])

AMPLITUDE_SATURATION = 65400
ADC = 65500
LOW_AMPLITUDE = 65300
MAX_PHASE = 30000.0
MAX_AMPLITUDE = 2047
GRAYSCALE_SATURATION = 4095
THRESHOLDED_PHASE = 0
THRESHOLDED_AMPL = 0
CAMERA_VERSION = ['espros','oyla_2_camera','oyla_1_camera_v0','oyla_1_camera_v1','oyla_1_camera_v2','oyla_1_camera_v3_cds','oyla_1_camera_v3_dus']
FOV  =[(90,70),(30,10),(44,33),(44,33),(44,33),(44,33),(44,33)]





def getComparedHDRAmplitudeImage_vectorized(amp,dist= None):
    ##########################################################################
    # needed for sampling from correct row in HDR distsance, based on amplitude values
    ##########################################################################
    highAmpThreshold = 2045
    Amp_case = np.where((amp[:,::2] > amp[:,1::2]), 1 , 2 )
     
    Amp_case = np.where(((amp[:,::2]>=highAmpThreshold )*(amp[:,1::2]<=highAmpThreshold)), 2 , Amp_case )
    
    Amp_case = np.where(((amp[:,::2]<=highAmpThreshold )*(amp[:,1::2]>=highAmpThreshold)), 1 , Amp_case )
    
    Amp_case = np.where(((amp[:,::2]<highAmpThreshold )*(amp[:,1::2]<highAmpThreshold)* (amp[:,::2] >= amp[:,1::2])), 1 , Amp_case )
    
    Amp_case = np.where(((amp[:,::2]<highAmpThreshold )*(amp[:,1::2]<highAmpThreshold)* (amp[:,::2] < amp[:,1::2])), 2 , Amp_case )
    
    amp[:,::2] = np.where((Amp_case==2), amp[:,1::2] , amp[:,::2])
    amp[:,1::2] = np.where((Amp_case==1), amp[:,::2] , amp[:,1::2])
    
    if dist is not None:
        dist[:,::2] = np.where(((Amp_case==2)*(dist is not None)), dist[:,1::2] , dist[:,::2])
        dist[:,1::2] = np.where(((Amp_case==1)*(dist is not None)), dist[:,::2] , dist[:,1::2])
    return amp, dist


def stitch_frame(frame1, frame2):
    
    frame = np.vstack((np.fliplr(frame1),np.flipud(frame2)))
    return frame

def phase_to_distance(img, ambiguity_distance = None, range_max = None, range_min = None):
    ##############################################
    # function to rescale phase data to distance and then thresholding with range max and min
    #############################################
    MAX_PHASE = 30000.0
    if ambiguity_distance:
        img = img/(MAX_PHASE)*ambiguity_distance
    #print(np.max(img)) 
    # Changed this so that values outside range can be set to a color
    # if range_max:
    #     img[np.where(img>range_max)] = 0
    # if range_min:
    #     img[np.where(img<range_min)] = 0
    #     #print("sum,max after range ",range_max, range_min ,np.sum(img),np.max(img))
    # return img
        
    if range_max and range_min:
        outside_range_indices = np.where(np.logical_and(np.logical_or(img>range_max,img<range_min),img>0))
    elif range_max:
        outside_range_indices = np.where(img>range_max)
    elif range_min:
        outside_range_indices = np.where(img<range_min)
    else:
        outside_range_indices = None
            
    return img, outside_range_indices


def transformation3_(IM, transform_types='cartesian',fov_angle=94,no_data_indices = None, range_max = None, range_min = None):
    '''
    To do spherical (i,j, rcm) into (x,y,z) where rcm is range in cm, i,j are pixel indices
    '''
    width = IM.shape[0]
    height = IM.shape[1]
    angle = fov_angle
    alfa0 = (angle * 3.14159265359) / 360.0;
    step = alfa0 / (width/2)
    X=[]
    Y=[]
    Z=[]

    dataPixelfield = IM
    for y in range(height):
        beta = (y-height/2) * step;
        for x in range(width):
            alfa = (x-width/2) * step;
            #rcm =  (speedOfLight*ModFreq*1000)*(dataPixelfield[x,y]/30000.0)/10.0;

            if(transform_types =='cartesian'):
                rcm = IM[x][y]
                X.append(rcm * np.cos(beta) * np.sin(alfa)+width/2)
                Y.append(rcm * np.sin(beta)+height/2)
                Z.append(rcm*np.cos(alfa)*np.cos(beta))
            else:
                X.append(x)
                Y.append(y)
                Z.append(IM[x][y])

    x = np.array(X)
    y = np.array(Y)
    z = np.array(Z)
    # remove points that are outside the range
    no_data_indices = None

    if no_data_indices:
        no_data_indices = np.ravel_multi_index(no_data_indices,(width,height))
        #print(no_data_indices)
        x = np.delete(x.flatten(),no_data_indices)
        y = np.delete(y.flatten(),no_data_indices)
        z = np.delete(z.flatten(),no_data_indices)
    else:
        if range_min:
            z_ = z.copy()
            x = x[z_ >= range_min]
            y = y[z_ >= range_min]
            z = z[z_ >= range_min]

        if range_max:
            z_ = z.copy()
            x = x[z_ < range_max]
            y = y[z_ < range_max]
            z = z[z_ < range_max]

    print('nv',np.histogram(z))
    return x, y, z



def transformation3(IM, transform_types='cartesian',fov_angle=94,fov_angle_o = None,no_data_indices = None):
    '''
    To do spherical (i,j, rcm) into (x,y,z) where rcm is range in cm, i,j are pixel indices
    Input IM (rcm) range data in cm 
    Output x, y, z in rcm in spherical coordinates. If cartesian transform then only transformation is removal of no_data_ind
    '''

    # if ambiguity_distance:
    #     IM_new = np.rint((int(ambiguity_distance)*(IM/MAX_PHASE)))
    IM_new = IM
    width = IM_new.shape[0]
    height = IM_new.shape[1]
    #print(width, height,fov_angle, fov_angle_o)
    # needed for sphericcal to cartesian
    alfa0 = fov_angle * np.pi/ 360.0;  
    step = 2*alfa0/width;
    if fov_angle_o is not None:
        beta0 = fov_angle_o * np.pi/ 360.0;  
        step_o = 2*beta0/height;
    else:
        step_o = step
    #print(width,height,fov_angle,fov_angle_o)
    beta_array = (np.resize(np.arange(height) - height/2,(width,height)))*step_o
    beta_array_cos = np.cos(beta_array)
    beta_array_sin = np.sin(beta_array)
    alpha_array = (np.resize(np.arange(width)- width/2,(height,width)).transpose() )*step
    alpha_array_cos = np.cos(alpha_array)
    alpha_array_sin = np.sin(alpha_array)

    #multipication by 100 to take into cm which is needed according to espros (must be something to do with focal length)
    if(transform_types =='cartesian'):
        x = np.multiply(np.multiply(IM_new,beta_array_cos),alpha_array_sin)+width/2
        y = np.multiply(IM_new,beta_array_sin)+height/2
        z = np.multiply(np.multiply(IM_new,beta_array_cos),alpha_array_cos)
    elif(transform_types == 'cartesian_no_shift'):
        x = np.multiply(np.multiply(IM_new,beta_array_cos),alpha_array_sin)
        y = np.multiply(IM_new,beta_array_sin)
        z = np.multiply(np.multiply(IM_new,beta_array_cos),alpha_array_cos)                
    elif(transform_types =='spherical'):
        x = np.resize(np.arange(width),(height,width)).transpose()## Y = y
        y = np.resize(np.arange(height),(width,height))## X  = x
        z = IM_new

    # till above full matrix, now removing only a select few
    if no_data_indices:
        no_data_indices = np.ravel_multi_index(no_data_indices,(width,height))
        x = np.delete(x.flatten(),no_data_indices)
        y = np.delete(y.flatten(),no_data_indices)
        z = np.delete(z.flatten(),no_data_indices)
        rcm = np.delete(IM_new.flatten(),no_data_indices)
    if False:
        #only for experiment
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        xyz = np.c_[x,y,z]
        pcd.points = o3d.utility.Vector3dVector(xyz)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
        out = np.setdiff1d(np.arange(xyz.shape[0]),ind)
        #print(len(out))
        x = np.delete(x,out)
        y = np.delete(y,out)
        z = np.delete(z,out)
        rcm = np.delete(rcm,out)
    #print('v',np.histogram(z))
    return x,y,z,rcm
def threshold_signals(phase, ampl, reflectivity_thresh, *argv):
    # curve fitted to 100% reflectivity -- see one of pinned post from ralph
   
    for arg in argv:
        phase[arg] = THRESHOLDED_PHASE
        ampl[arg] = THRESHOLDED_AMPL
    return phase, ampl, ind

def threshold_coordinates(x, y, z,rcm, range_max = None, range_min = None, x_max = None, x_min = None, y_max = None, y_min = None, img = None):
    '''
    thresholding x,y,z using range for each
    '''
    if img is not None:
        _img = img.flatten()
        flat_img_indices = np.where(_img>0)[0]
        print(len(x), len(flat_img_indices))
        assert len(flat_img_indices) == len(x)
        
    if range_min is not None:
        _m = rcm.copy()
        l = range_min
        x = x[_m >= l]
        y = y[_m >= l]
        z = z[_m >= l]
        rcm = rcm[_m >= l]
        if img is not None:
            flat_img_indices = flat_img_indices[_m>=l]
        
    if range_max is not None:
        _m = rcm.copy()
        l = range_max
        x = x[_m < l]
        y = y[_m < l]
        z = z[_m < l]
        rcm = rcm[_m < l]
        if img is not None:
            flat_img_indices = flat_img_indices[_m<l]
            
    if x_min is not None:
        _m = x.copy()
        l = x_min
        x = x[_m >= l]
        y = y[_m >= l]
        z = z[_m >= l]
        rcm = rcm[_m >= l]
        if img is not None:
            flat_img_indices = flat_img_indices[_m>=l]
            
    if x_max is not None:
        _m = x.copy()
        l = x_max
        x = x[_m < l]
        y = y[_m < l]
        z = z[_m < l]
        rcm = rcm[_m < l]
        if img is not None:
            flat_img_indices = flat_img_indices[_m<l]
            
    if y_min is not None:
        _m = y.copy()
        l = y_min
        x = x[_m >= l]
        y = y[_m >= l]
        z = z[_m >= l]
        rcm = rcm[_m >= l]
        if img is not None:
            flat_img_indices = flat_img_indices[_m>=l]
            
    if y_max is not None:
        _m = y.copy()
        l = y_max
        x = x[_m < l]
        y = y[_m < l]
        z = z[_m < l]
        rcm = rcm[_m < l]
        if img is not None:
            flat_img_indices = flat_img_indices[_m<l]
    if img is not None:
        __img = np.zeros_like(_img)
        __img[flat_img_indices] = _img[flat_img_indices]
        img = __img.reshape(img.shape)
        indices = np.where(np.logical_and(img == 0, _img.reshape(img.shape) !=0))
        return x, y, z, rcm, img, indices
    return x, y, z, rcm



def camera_calibrations(rgb,camera_version = 'oyla_2_camera',depth = None, ampl = None):
    ##########################################################################
    # 
    #Assuming that data is depth+amplitude specifically for new camera
    #And after alll the work that Ralph has put in overlaying depth map on rgbimage (see ppt)
    # C:\Users\rwcsp\Oyla Dropbox\Oyla\02 Technical\04 Calibration
    ##########################################################################
    _depth = depth
    _ampl = ampl
    _rgb = rgb

    if _rgb is not None:
        i = np.argmax(_rgb.shape)
        if i != 1:
            _rgb = np.transpose(_rgb,(1,0,2))
    if _depth is not None:
        assert _depth.shape == _ampl.shape
        i = np.argmax(_depth.shape)
        if i != 1:
            if _depth.ndim==2:
                _depth = np.transpose(_depth,(1,0))
                _ampl = np.transpose(_ampl,(1,0))
            else:
                _depth = np.transpose(_depth,(1,0,2))
                _ampl = np.transpose(_ampl,(1,0,2))

    if camera_version == 'oyla_2_camera':
        if _rgb is not None:
            assert _rgb.shape ==(480,640,3)
            _rgb = np.fliplr(np.flipud(_rgb))
            _rgb = _rgb[140:352,36:601,:]
        #print(_depth.shape,_rgb.shape)        
    elif camera_version == 'oyla_1_camera_v0':
        if _rgb is not None and _rgb.shape ==(480,640,3):
            _rgb = np.fliplr(_rgb)
            _rgb = _rgb[:,77:578]
        if _depth is not None :
            if _depth.shape == (120,160):
                _depth = cv2.resize(_depth,(320,240), interpolation = cv2.INTER_NEAREST)
                _ampl = cv2.resize(_ampl,(320,240), interpolation = cv2.INTER_NEAREST)
            assert _depth.shape == (240,320)
            _depth = np.flipud(_depth)
            _ampl = np.flipud(_ampl)
            _depth = _depth[18:220,25:302]
            _ampl = _ampl[18:220,25:302]
    elif camera_version == 'oyla_1_camera_v1':
        if _rgb is not None and _rgb.shape ==(360,640,3):
            _rgb = np.fliplr(_rgb)
            _rgb = _rgb[:,80:570]
        if _depth is not None :
            if _depth.shape == (120,160):
                _depth = cv2.resize(_depth,(320,240), interpolation = cv2.INTER_NEAREST)
                _ampl = cv2.resize(_ampl,(320,240), interpolation = cv2.INTER_NEAREST)
            assert _depth.shape == (240,320)
            _depth = np.flipud(_depth)
            _ampl = np.flipud(_ampl)
            _depth = _depth[21:219,27:302]
            _ampl = _ampl[21:219,27:302]
    elif camera_version == 'oyla_1_camera_v2':
        if _rgb is not None and _rgb.shape ==(360,640,3):
            _rgb = np.fliplr(_rgb)
            _rgb = _rgb[:,77:597]
        if _depth is not None :
            if _depth.shape == (120,160):
                _depth = cv2.resize(_depth,(320,240), interpolation = cv2.INTER_NEAREST)
                _ampl = cv2.resize(_ampl,(320,240), interpolation = cv2.INTER_NEAREST)
            if _depth.shape == (240,320):
                _depth = np.flipud(_depth)
                _ampl = np.flipud(_ampl)
                _depth = _depth[20:220,26:312]
                _ampl = _ampl[20:220,26:312]
    elif 'oyla_1_camera_v3' in camera_version:
        print('version 3')
        if camera_version == 'oyla_1_camera_v3_cds':
            rgb_scale = 1/1.84
            depth_scale = None
            rgb_translate = np.asarray([1,2])
            depth_translate = np.zeros_like(rgb_translate)
            fx = 2400/5.52
            fy = 2400/5.52
            ox = 50
            oy = 14
            K=-9e-1
        elif camera_version == 'oyla_1_camera_v3_dus':
            rgb_scale = None
            depth_scale = 1.84
            rgb_translate = np.asarray([3,3])
            depth_translate = np.zeros_like(rgb_translate)
            fx=2400/3
            fy = 2400/3
            ox = 93
            oy = 40
            K=-8e-1
         
        if _rgb is not None and _rgb.shape ==(360,640,3):
            _rgb = np.fliplr(_rgb)
            if rgb_scale is not None:
                _rgb = cv2.resize(_rgb,None,fx=rgb_scale,fy=rgb_scale)#, interpolation = cv2.INTER_NEAREST)

            h, w = _rgb.shape[:2]
            cx = w//2
            cy = h//2
            KK_new=np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]])
            ox = ox+cx
            oy = oy+cy

            #R = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
    
            #urgb = np.ones_like(_rgb)
            #for i in range(0,3):
            #   urgb[:,:,i] = undistort(_rgb[:,:,i],fx,fy,cx,cy,K4=K,oy=oy,ox=ox)
            #_rgb = urgb
            _rgb = undistort(_rgb,fx,fy,cx,cy,K4=K,oy=oy,ox=ox)
            
        if _depth is not None :
            _depth = np.flipud(_depth)
            _ampl = np.flipud(_ampl)
            if _depth.shape[:2] == (120,160):
                if depth_scale is None:
                    depth_scale = 1
                _depth = cv2.resize(_depth,None,fx=2*depth_scale,fy=2*depth_scale)#, interpolation = cv2.INTER_NEAREST)
                _ampl = cv2.resize(_ampl,None,fx=2*depth_scale,fy=2*depth_scale)#, interpolation = cv2.INTER_NEAREST)
            depth_centeriod = np.asarray(_depth.shape[:2])//2
            rgb_centeriod = np.asarray(_rgb.shape[:2])//2

            # assume that translates are correct axis -- x axis horizontal with shift to right positive, y axis vert with shift to down negative
            # I will also have to check what is the axis here 
            rgb_ul = (-1,1)*np.asarray(rgb_centeriod) + rgb_translate
            rgb_br = (1,-1)*np.asarray(rgb_centeriod) + rgb_translate

            depth_ul = (-1,1)*np.asarray(depth_centeriod) + depth_translate
            depth_br = (1,-1)*np.asarray(depth_centeriod) + depth_translate

            ul = np.zeros_like(rgb_ul)
            ul[0] = np.max((rgb_ul[0],depth_ul[0]))
            ul[1] = np.min((rgb_ul[1],depth_ul[1]))
            br = np.zeros_like(rgb_br)
            br[0] = np.min((rgb_br[0],depth_br[0]))
            br[1] = np.max((rgb_br[1],depth_br[1]))

            _ul = np.abs(rgb_ul-ul)
            _w,_h =br -ul
            _h = np.abs(_h)
            #print(_ul,rgb_ul,ul,_w,_h,rgb_br,br)
            _rgb = _rgb[_ul[0]:_ul[0]+_w,_ul[1]:_ul[1]+_h,:]
            #print(_rgb.shape)
            
            _ul = np.abs(depth_ul-ul)
            _w,_h =br -ul
            _h = np.abs(_h)
            #print(_depth.shape)
            if _depth.ndim == 2:
                _depth = _depth[_ul[0]:_ul[0]+_w,_ul[1]:_ul[1]+_h]
                _ampl = _ampl[_ul[0]:_ul[0]+_w,_ul[1]:_ul[1]+_h]
            else:
                _depth = _depth[_ul[0]:_ul[0]+_w,_ul[1]:_ul[1]+_h,:]
                _ampl = _ampl[_ul[0]:_ul[0]+_w,_ul[1]:_ul[1]+_h,:]
            #print(_depth.shape)
    elif camera_version == 'espros':
            if _depth is not None:
                _depth = np.flipud(np.fliplr(_depth))
                _ampl = np.flipud(np.fliplr(_ampl))
    print(_rgb.shape,_depth.shape,_ampl.shape)           
    return _rgb, _depth,_ampl

def _write_files(stored_data,c,save_path,fno,_fp,__fp):
    scipy.io.savemat(save_path+'/data_'+str(c)+'_'+str(fno)+'.mat',{'data':stored_data})
    if len(stored_data)>5:
        __fp.write(str(stored_data[5])+'\n')
    # with open(save_path+'/imageDistance_'+str(c)+'_'+str(fno)+'.bin','wb') as fp:
    #     for k in range(stored_data[2][0][0].shape[2]-1):
    #         _tmp = bytearray(stored_data[2][0][0][:,:,k].transpose())
    #         fp.write(_tmp)
    k = stored_data[2][0][0].shape[2]-1

    # with open(save_path+'/imageDistance_'+str(c)+'_'+str(fno)+'_ampl.bin','wb') as fp:
    #     _tmp = bytearray(stored_data[2][0][0][:,:,k].transpose())
    #     fp.write(_tmp)
    _fp.write('imageDistance_'+str(c)+'_'+str(fno)+'\n')
    
def write_files(stored_data,c,save_path,fno=None):
    ##########################################################################
    #write data to file
    ##########################################################################
    _fp = open(save_path+'/imageDistance_'+str(c)+'.idx','a')
    __fp = open(save_path+'/timestamps.txt','a')
    if fno is None:
        for fno in stored_data.keys():
            _write_files(stored_data[fno],c,save_path,fno,_fp,__fp)
    else:
        _write_files(stored_data,c,save_path,fno,_fp,__fp)
    _fp.close()
    __fp.close()
    
# for fno in stored_data[c].keys():
#     #scipy.io.savemat(save_path+'/data_'+str(c)+'_'+str(fno)+'.mat',{'data':stored_data[c][fno]})
#     with open(save_path+'/imageDistance_'+str(c)+'_'+str(fno)+'.bin','wb') as fp:
#         for k in range(stored_data[c][fno].shape[2]-1):
#             _tmp = bytearray(stored_data[c][fno][:,:,k].transpose())
#             fp.write(_tmp)
#     k = stored_data[c][fno].shape[2]-1

#     with open(save_path+'/imageDistance_'+str(c)+'_'+str(fno)+'_ampl.bin','wb') as fp:
#         _tmp = bytearray(stored_data[c][fno][:,:,k].transpose())
#         fp.write(_tmp)
#     _fp[c].write('imageDistance_'+str(c)+'_'+str(fno)+'\n')
# _fp[c].close()


def threshold_filter(raw_phase, raw_ampl, reflectivity_thresh, range_max, range_min, ampl_min, filter_params, ambiguity_distance, qp_phase = None, qp_ampl = None):
    indices = {}
    
    if True:
        # All theses indices are on original data 
        indices['amplitude_saturated'] = np.where(raw_ampl>=AMPLITUDE_SATURATION)
        indices['amplitude_low'] = np.where(raw_ampl==LOW_AMPLITUDE)
        

        indices['amplitude_beyond_range'] = np.where(np.logical_or(raw_ampl < ampl_min,
                                                                   np.logical_and(raw_ampl>MAX_AMPLITUDE,raw_ampl<LOW_AMPLITUDE)))
    
        phase_min = range_min*MAX_PHASE/ambiguity_distance
        phase_max = range_max*MAX_PHASE/ambiguity_distance
        indices['phase_beyond_range'] = np.where(np.logical_or(raw_phase<phase_min, raw_phase>phase_max))

        thresholded_phase = raw_phase.copy()
        thresholded_ampl = raw_ampl.copy()
        # Here original data is converted to data -- we are forcing them to go to a certain value, but the correct formulation should have been to ignore them.
        # So after this step everything is "signal" ie its not no_data_indices
        for k in indices.keys():
            ind = indices[k]
            thresholded_phase[ind] = THRESHOLDED_PHASE
            thresholded_ampl[ind] = THRESHOLDED_AMPL

        if qp_phase is not None:
            
            thresholded_phase = np.floor(thresholded_phase.astype('float32')/qp_phase+0.5)*qp_phase
            thresholded_phase = thresholded_phase.astype('float32')
            
        if qp_ampl is not None:
            thresholded_ampl = np.floor(thresholded_ampl.astype('float32')/qp_ampl+0.5)*qp_ampl
            thresholded_ampl = thresholded_ampl.astype('float32')

        if qp_phase is not None or qp_ampl is not None:
            ind = np.where(np.logical_or(np.logical_or(thresholded_phase<phase_min, thresholded_phase>phase_max),np.logical_or(thresholded_ampl<ampl_min, thresholded_ampl>MAX_AMPLITUDE)))
            thresholded_phase[ind] = THRESHOLDED_PHASE
            thresholded_ampl[ind] = THRESHOLDED_AMPL
            indices['quantized_beyond_range'] = ind
        
        if reflectivity_thresh:
            t = 141201*np.power(thresholded_ampl,-0.492)
            ind = np.where(thresholded_phase>t)
            thresholded_phase[ind] = THRESHOLDED_PHASE
            thresholded_ampl[ind] = THRESHOLDED_AMPL
            indices['reflectivity_beyond_range'] = ind

                    
        thresholded_phase = thresholded_phase.astype('float32')/phase_max
        guide = thresholded_ampl.astype('float32')/MAX_AMPLITUDE
        filtered_phase, filtered_ampl = filters_in_use(thresholded_phase, guide, filter_params,phase_min,phase_max, ambiguity_distance)
        filtered_phase *= phase_max #
        filtered_ampl *= MAX_AMPLITUDE
        # this is no_data_indices because everything else is valid signal

        ind = np.where(np.logical_or(np.logical_or(filtered_phase<phase_min, filtered_phase>phase_max),np.logical_or(filtered_ampl<ampl_min, filtered_ampl>MAX_AMPLITUDE)))
        
        filtered_phase[ind] = THRESHOLDED_PHASE
        filtered_ampl[ind] = THRESHOLDED_AMPL
        indices['filtered_beyond_range'] = ind

    else:
        thresholded_phase = raw_phase
        thresholded_ampl = raw_ampl.astype('float32')/np.max(raw_ampl)
        filtered_phase = raw_phase.astype('float32')/np.max(raw_phase)

                
                

    
    return filtered_phase, filtered_ampl, indices

def rgb_to_srgb(rgb):
    ret = np.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = np.power(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret

def undistort(I,fx,fy,cx,cy,K4,ox=None,oy=None,R = None):
    [nr,nc] = I.shape[:2]
    #Irec = np.ones_like(I);
    ifx = 1/fx
    ify = 1/fy
    if R is None:
        R = np.eye(3)
    #import time
    #start = time.time()
    [mx,my] = np.meshgrid(np.arange(0,nc), np.arange(0,nr))
    #print('a',time.time()-start)
    
    px = mx.reshape(-1,1)
    py = my.reshape(-1,1)
    rays = np.hstack(((px ),(py ),np.ones(px.shape)));

    _px = ifx*((px)-cx)
    _py = ify*((py)-cy)
    _pz = np.ones_like(_py)
    
    rays =  np.hstack((_px,_py,_pz))
    rays2 = np.dot(R,rays.T)
    x = np.vstack((rays2[0,:]/rays2[2,:],rays2[1,:]/rays2[2,:]))
    #x = np.hstack((_px,_py)).T
    
    offset = [0,0]
    if ox is not None and oy is not None:
        ind = np.where((py==oy) & (px==ox))
        #print(ind[0])
        offset = x[:,ind[0]]
    
    r2 = (x[0,:]-offset[0])**2+(x[1,:]-offset[1])**2
    cdist = 1+K4*(r2**2)
    xd1 = x * (np.ones((2,1))*cdist)
    #print('K4',K4,offset)
    
    px2 = xd1[0,:]*fx+cx#p2[0,:]
    py2 = xd1[1,:]*fy+cy#p2[1,:]
    
    px_0 = np.floor(px2)
    py_0 = np.floor(py2)
    
    good_points = np.where((px_0 >= 0) & (px_0 < (nc-1)) & (py_0 >= 0) & (py_0 < (nr-1)))
    
    px2 = px2[good_points]
    py2 = py2[good_points]
    px_0 = px_0[good_points]
    py_0 = py_0[good_points]
    
    alpha_x = px2 - px_0;
    alpha_y = py2 - py_0;
    a1 = (1 - alpha_y)*(1 - alpha_x);
    a2 = (1 - alpha_y)*alpha_x;
    a3 = alpha_y * (1 - alpha_x);
    a4 = alpha_y * alpha_x;
    
    ind_lu = py_0 * nc + px_0 ;
    ind_ru = (py_0 + 1) * nc + px_0;
    ind_ld = py_0 * nc + (px_0 +1) ;
    ind_rd = (py_0 + 1) * nc + (px_0 +1) ;
    ind_new = np.squeeze((py[good_points])*nc + px[good_points])

    JJ = np.zeros_like(I)
    for i in range(0,3):
        II = I[:,:,i].flatten()
        J = np.ones_like(II)
        J[np.squeeze(ind_new).astype('int')] = a1 * II[ind_lu.astype('int')] + a3* II[ind_ru.astype('int')] + a2 * II[ind_ld.astype('int')] + a4 * II[ind_rd.astype('int')]
        JJ[:,:,i] = J.reshape(I.shape[:2])


    #print('ee',time.time()-start)
    # start = time.time()
    # ind_lu = np.unravel_index(ind_lu.astype('int'),I.shape[:2])
    # ind_ru = np.unravel_index(ind_ru.astype('int'),I.shape[:2])
    # ind_ld = np.unravel_index(ind_ld.astype('int'),I.shape[:2])
    # ind_rd = np.unravel_index(ind_rd.astype('int'),I.shape[:2])
    # ind_new = np.unravel_index(np.squeeze(ind_new),I.shape[:2])
    # print('f',time.time()-start)
    # start = time.time()
    # Irec[ind_new] = a1 * I[ind_lu] + a3* I[ind_ru] + a2 * I[ind_ld] + a4 * I[ind_rd]
    # print('g',time.time()-start)
    #Irec[Irec>1] = 1
    #Irec[Irec<0] = 0
    #print(np.mean(np.abs(Irec-I))*255,np.mean(Irec),np.mean(I),np.min(Irec)*255)
    return JJ
