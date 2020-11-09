import cv2
import numpy as np
from scipy.signal import convolve2d
from oyla.enhancement.homofilt import HomomorphicFilter

def fit_reflectivity(ampl,phase,camera_version='oyla_2_camera'):
    if camera_version == 'oyla_2_camera':
        reflectivity = phase*phase*np.power(ampl,0.986)/(2.1*np.power(10,10))
    elif 'oyla_1_camera' in camera_version:
        reflectivity = phase*phase*np.power(ampl,0.996)/(5.63*np.power(10,10))
    return reflectivity
def rgb_equalize_histogram(rgb):
    rgb = (rgb*255).astype('uint8')
    hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv)
    _v = cv2.equalizeHist(v)
    hsv = cv2.merge((h,s,_v))
    ergb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    ergb[ergb>255] = 255
    ergb[ergb<0] = 0
    ergb = ergb.astype('float32')/255.0
    return ergb

def Stransform(x,delta1=0,delta2=1,n=2,m=0.5):
    #https://www.isical.ac.in/~sarif_r/papers/naik-12tip12-hue.pdf
    y = np.zeros_like(x)
    
    ind = np.where((x<=m)&(x>=delta1))
    if type(m)==np.ndarray:
        y[ind] = delta1+(m[ind]-delta1)*np.power((x[ind]-delta1)/(m[ind]-delta1+0.00001),n)
    elif type(m)==float:
        y[ind] = delta1+(m-delta1)*np.power((x[ind]-delta1)/(m-delta1+0.00001),n)
    ind = np.where((x>m)&(x<=delta2))
    if type(m)==np.ndarray:
        y[ind] = delta2-(delta2-m[ind])*np.power((delta2-x[ind])/(delta2-m[ind]+0.00001),n)
    elif type(m)==float:
        y[ind] = delta2-(delta2-m)*np.power((delta2-x[ind])/(delta2-m+0.00001),n)
    #print(np.where(y>1))
    y[y>1] = 1
    y[y<0] = 0
    return y




def clean_algorithm(reflectivity):
    residual = np.array(reflectivity)
    psf = cv2.getGaussianKernel(3,0.5)
    psf = np.matmul(psf,psf.transpose())
    c=0
    while True:
        if np.max(residual)>1:
            source = np.zeros_like(reflectivity)
            source[np.unravel_index(np.argmax(residual),residual.shape)] = 1
            c += 1
            o = convolve2d(source,psf,mode='same')
            residual = residual*(1-o)
        else:
            break
    return residual

def clean_algorithm2(reflectivity):
    residual = np.array(reflectivity)
    psf = cv2.getGaussianKernel(3,0.5)
    psf = np.matmul(psf,psf.transpose())
    c = 0
    while True:
        if np.count_nonzero(residual>1)>0:
            source = np.zeros_like(reflectivity)
            source[residual>1] = 1.0/np.count_nonzero(residual>1)
            #print('x')
            c += 1
            o = convolve2d(source,psf,mode='same')
            residual = residual*(1-o)
        else:
            break

    return residual

def supress_saturation(reflectivity,indices):
    oyla_refl_saturated = np.array(reflectivity)
    for i in range(indices['amplitude_saturated'][0].shape[0]):
        x = indices['amplitude_saturated'][0][i]
        y = indices['amplitude_saturated'][1][i]
        sx = np.max([x-3,0])
        sy = np.max([y-3,0])
        ex = np.min([x+3,reflectivity.shape[0]])
        ey = np.min([y+3,reflectivity.shape[1]])
        #print(oyla_refl_saturated[sx:ex,sy:ey])
        oyla_refl_saturated[sx:ex,sy:ey] = 0
        #print(oyal_refl_saturated[sx:ex,sy:ey])
    #oyla_refl_saturated[oyla_refl_saturated>np.max(rgb_refl)] = 0
    #oyla_refl_saturated[oyla_refl_saturated>1] = 1
    return oyla_refl_saturated

def do_reflectivity_enhancement(rgb_img, filtered_phase, thresholded_ampl, indices, camera_version = 'oyla_1_camera_v3'):
    homo_filter = HomomorphicFilter()
    
    rgb_img = rgb_img.astype('float32')/255.0
    hsv_img = cv2.cvtColor(rgb_img,cv2.COLOR_RGB2HSV)
    print("doing reflectivity enhancement")
    
    oyla_reflectivity = fit_reflectivity(thresholded_ampl, filtered_phase, camera_version=camera_version)
    h,s,v = cv2.split(hsv_img)
    illum,refl = homo_filter.get_illumination_reflectance(I=v,filter_params=[30,2])
    scale = np.max(refl)
    #print('nonzero ampl',np.count_nonzero(thresholded_ampl))
    #print('nonzero phase',np.count_nonzero(filtered_phase))
    #print('nonzero refl',np.count_nonzero(oyla_reflectivity))
    oyla_reflectivity = supress_saturation(oyla_reflectivity, indices)
    oyla_reflectivity[oyla_reflectivity>2] = 0
    oyla_reflectivity[oyla_reflectivity>1] = 1
    #print('mean ref',np.mean(oyla_reflectivity))
    mi = np.min(refl)
    ma = np.max(refl)
    refl = (refl-mi)/(ma-mi)
    scale = 1
    ind = np.where((refl/scale<(oyla_reflectivity)))
    refl[ind] = (oyla_reflectivity)[ind]*scale
    refl = refl*(ma-mi)+mi
    _v = illum*refl-1
    _v = _v.astype('float32')
    _v[_v>1]=1
    ehsv_img = cv2.merge((h,s,_v))
    ergb_img = cv2.cvtColor(ehsv_img,cv2.COLOR_HSV2RGB)
    ergb_img[ergb_img>1] = 1
    ergb_img[ergb_img<0] = 0
    
    return ergb_img
