import matplotlib.pyplot as plt
import cv2
import numpy as np

def undistort(I,fx,fy,cx,cy,K4,ox=None,oy=None,R = None):
    [nr,nc] = I.shape[:2]
    Irec = np.ones_like(I);
    ifx = 1/fx
    ify = 1/fy
    if R is None:
        R = np.eye(3)
    
    [mx,my] = np.meshgrid(np.arange(0,nc), np.arange(0,nr))
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
        print(ind[0])
        offset = x[:,ind[0]]
    
    r2 = (x[0,:]-offset[0])**2+(x[1,:]-offset[1])**2
    cdist = 1+K4*(r2**2)
    xd1 = x * (np.ones((2,1))*cdist)
    print('K4',K4,offset)
   
    
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
    ind_new = (py[good_points])*nc + px[good_points];

    ind_lu = np.unravel_index(ind_lu.astype('int'),I.shape[:2])
    ind_ru = np.unravel_index(ind_ru.astype('int'),I.shape[:2])
    ind_ld = np.unravel_index(ind_ld.astype('int'),I.shape[:2])
    ind_rd = np.unravel_index(ind_rd.astype('int'),I.shape[:2])
    ind_new = np.unravel_index(np.squeeze(ind_new),I.shape[:2])
   
    Irec[ind_new] = a1 * I[ind_lu] + a3* I[ind_ru] + a2 * I[ind_ld] + a4 * I[ind_rd]
    #Irec[Irec>1] = 1
    #Irec[Irec<0] = 0
    print(np.mean(np.abs(Irec-I))*255,np.mean(Irec),np.mean(I),np.min(Irec)*255)
    return Irec

if __name__ == "__main__":
    import sys
    rgb = plt.imread(sys.argv[1])
    depth = plt.imread(sys.argv[2])
    resize = True
    if resize:
        rgb = cv2.resize(rgb,None,fx=1/1.84,fy=1/1.84, interpolation = cv2.INTER_NEAREST)
        depth = cv2.resize(depth,(320,240),interpolation = cv2.INTER_NEAREST)
        fx = 2400/5.52
        fy = 2400/5.52
        ox = 51
        oy = 14
        K=-4.21e-1
    else:
        #assume depth is 160x120 and rescaled to 2*160*1.84
        depth = cv2.resize(depth,None,fx=2*1.84,fy=2*1.84, interpolation = cv2.INTER_NEAREST)
        fx=2400/3
        fy = 2400/3
        ox = 93
        oy = 25
        K=-3.72e-1*2
        
    h, w = rgb.shape[:2]
    print(w//2,h//2)
    cx = w//2
    cy = h//2
    KK_new=np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]])
    ox = ox+cx
    oy = oy+cy

    #R = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
    
    urgb = np.ones_like(rgb)
    for i in range(0,3):
        print(i)
        urgb[:,:,i] = undistort(rgb[:,:,i],fx,fy,cx,cy,K4=K,oy=oy,ox=ox)

    plt.imsave('rgb.png',urgb)
    plt.imsave('depth.png',depth)
