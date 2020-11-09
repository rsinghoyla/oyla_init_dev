import cv2
import numpy as np
from skimage.segmentation import felzenszwalb
try:
        import open3d as o3d
except ImportError:
        o3d=None
def wrapper_around_segmentation(img,img_type,min_size,sigma,scale,counts ):
        if img.ndim == 3:
                img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        _img = img.astype('float32')/np.max(img+0.000001)
        _felz_segs = felzenszwalb(_img,multichannel=False,min_size=min_size,sigma=sigma,scale = scale)
        print(img_type+' Number of segments',np.unique(_felz_segs).shape[0])
        if img_type not in counts:
                counts[img_type] = []
        counts[img_type].append(np.unique(_felz_segs).shape[0])
        _felz_segs  = _felz_segs.astype('float32')/np.max(_felz_segs+0.000001)
        _felz_segs = (_felz_segs*255).astype('uint8')
        felz_segs_img = cv2.cvtColor(_felz_segs,cv2.COLOR_GRAY2RGB)
        return felz_segs_img, counts



def depth_to_xyz_view_img2(pcd,x_max, x_min, y_max, y_min, z_max, z_min,  view = 'yz', output_width = None, output_height = None, factor = 1, isometric_R = None):
    # Input:
    #   pcd: (N', 4)
    # Output:
    #   birdview: (w, l, 3)
    if isinstance(pcd, np.ndarray):
        points = pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    #else:
    #
    points = np.asarray(pcd.points)
        
    if isometric_R is not None:
        pcd.rotate(isometric_R,center=pcd.get_center())
    #X_MAX, Y_MAX,Z_MAX = np.max(np.asarray(points),axis=0)
    #X_MIN, Y_MIN,Z_MIN = np.min(np.asarray(points),axis=0)
    #print(X_MAX,X_MIN,Y_MAX,Y_MIN,Z_MAX,Z_MIN)
    #print(x_max,x_min,y_max,y_min,z_max,z_min)
    if view =='yz':
        dim = [(y_max-y_min),(z_max-z_min)]
    if view =='xy':
        dim = [(x_max-x_min),(y_max-y_min)]
    if view == 'xz':
        dim = [(x_max-x_min),(z_max-z_min)]

    if output_width is None:
        VOXEL_A_SIZE = 2
        VOXEL_B_SIZE = 2
        output_width = int(dim[0] / VOXEL_A_SIZE)
        output_height = int(dim[1] / VOXEL_B_SIZE)
    else:
        VOXEL_A_SIZE = dim[0]/output_width
        VOXEL_B_SIZE = dim[1]/output_height
    #print(output_height,output_width,VOXEL_A_SIZE)
    birdview = np.zeros((output_height , output_width ))
    count = np.zeros_like(birdview)
    
    #factor = 3.1
    
    for point in np.asarray(points):
        x,y,z = point[0:3]
        _t = z
        #if X_MIN < x < X_MAX and cfg.Y_MIN < y < cfg.Y_MAX:
        if view == 'yz':
                a, b = int((y - y_min) / VOXEL_A_SIZE *factor), int((z - z_min) / VOXEL_B_SIZE * factor)
        elif view =='xy':
                a, b = int((x - x_min) / VOXEL_A_SIZE *factor), int((y - y_min) / VOXEL_B_SIZE * factor)
        elif view == 'xz':
                a, b = int((x - x_min) / VOXEL_A_SIZE *factor), int((z - z_min) / VOXEL_B_SIZE * factor)

        if a<=0:
            a = 1
        if b<0:
            b = 0
        if b> output_height-1:
            b = output_height-1
        if a>output_width-1:
            a = output_width
        #print(x,y,a,b)
        birdview[b, output_width-a] += _t
        count[b, output_width-a] += 1
    birdview = birdview/(count+0.000001)

    #divisor = np.max(birdview) - np.min(birdview)
    #birdview = birdview - np.min(birdview)
    # TODO: adjust this factor
    # birdview = np.clip((birdview / divisor * 255) *
    #                    5 * factor, a_min=0, a_max=255)
    #birdview = np.tile(birdview, 3).astype(np.uint8)

    return birdview

def depth_to_xyz_view_img(pcd,  a_max, a_min, b_max, b_min, view = 'yz',output_width = None, output_height = None, factor = 1, labels = None):
    # Input:
    #   pcd: (N', 4)
    # Output:
    #   birdview: (w, l, 3)
    #Z_MAX, X_MAX,Y_MAX = np.max(np.asarray(pcd.points),axis=0)
    #Z_MIN, X_MIN,Y_MIN = np.min(np.asarray(pcd.points),axis=0)
    if isinstance(pcd, np.ndarray):
        points = pcd
    else:
        points = np.asarray(pcd.points)
    
    if output_width is None:
        VOXEL_A_SIZE = 16
        VOXEL_B_SIZE = 16
        output_width = int((a_max - a_min) / VOXEL_A_SIZE)
        output_height = int((b_max - b_min) / VOXEL_B_SIZE)
    else:
        VOXEL_A_SIZE = (a_max - a_min)/output_width
        VOXEL_B_SIZE = (b_max - b_min)/output_height
    birdview = np.zeros(
        (output_height * factor+1, output_width * factor+1))
    
    for I, point in enumerate(points):
        x,y,z = point[0:3]
        #if X_MIN < x < X_MAX and cfg.Y_MIN < y < cfg.Y_MAX:
        if view == 'yz':
                if y<a_min:
                        print("lower than y_min",y)
                        continue
                if y>a_max:
                        print("higer than y_max",y)
                        continue
                if z>b_max:
                        continue
                if z<b_min:
                        continue
                a, b = int((y - a_min) / VOXEL_A_SIZE *factor), int((z - b_min) / VOXEL_B_SIZE * factor)
                
                        
        elif view =='xy':
                a, b = int((x - a_min) / VOXEL_A_SIZE *factor), int((y - b_min) / VOXEL_B_SIZE * factor)
        elif view == 'xz':
                a, b = int((x - a_min) / VOXEL_A_SIZE *factor), int((z - b_min) / VOXEL_B_SIZE * factor)
        if labels is None:
            birdview[b, output_width-a] += 1
        else:
            birdview[b, output_width-a] = labels[I]
    
    if labels is None:        
        birdview = birdview - np.min(birdview)
        divisor = np.max(birdview) - np.min(birdview)
        # TODO: adjust this factor
        birdview = np.clip((birdview / divisor * 255) *
                           5 * factor, a_min=0, a_max=255)
    #birdview = np.tile(birdview, 3).astype(np.uint8)

    return birdview, VOXEL_A_SIZE*factor, VOXEL_B_SIZE*factor

def perspective_projection(pcd, type_of_projection, row_max, row_min, col_max, col_min, num_cols = None, num_rows = None, voxel_row_size = None,
                            voxel_col_size=None, isometric_R = None, labels = None):
        # Input:
        #   pcd: (N', 4)
        # Output:
        #   birdview: (w, l, 3)
        if isinstance(pcd, np.ndarray):
                points = pcd
                if o3d is not None and isometric_R is not None:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points)
                        points = np.asarray(pcd.points)
        else:
                points = np.asarray(pcd.points)
    
        if isometric_R is not None:
                pcd.rotate(isometric_R,center=pcd.get_center())
    
 
        dim = [(row_max-row_min), (col_max-col_min)]

        if num_rows is None and voxel_row_size is not None:
                num_rows = int(dim[0]/voxel_row_size)
        else:
                voxel_row_size = dim[0]/num_rows
        if num_cols is None and voxel_col_size is not None:
                num_cols = int(dim[1]/ voxel_col_size)
        else:
                voxel_col_size = dim[1]/num_cols
        #print(output_height,output_width,VOXEL_A_SIZE)
        projection = np.zeros((num_rows , num_cols ))
        count = np.zeros_like(projection)
    
        #factor = 3.1
        print('abc',np.max(np.asarray(points),0),np.min(np.asarray(points),0))
        for index, point in enumerate(np.asarray(points)):
                x,y,z = point[0:3]
                
                if type_of_projection == 'yz':
                        a, b = int((y - col_min) / voxel_col_size), int((z - row_min) / voxel_row_size )
                        _t = x
                elif type_of_projection =='xy':
                        a, b = int((x - col_min) / voxel_col_size ), int((y - row_min) / voxel_row_size )
                        _t = z
                elif type_of_projection == 'xz':
                        a, b = int((x - col_min) / voxel_col_size ), int((z - row_min) / voxel_row_size )
                        _t = y

                if a<=0:
                    a = 1
                if b<0:
                    b = 0
                if b> num_rows-1:
                    b = num_rows-1
                if a> num_cols-1:
                    a = num_cols
                #print(x,y,a,b)
                if labels is None:
                        projection[b, num_cols-a] += _t
                        count[b, num_cols-a] += 1
                else:
                        projection[b,num_cols-a] = labels[index]
                        count[b,num_cols-a] = 1
        projection = projection/(count+0.000001)

        
        return projection

def depth_to_xyz_coords(pcd,  a_max, a_min, b_max, b_min, view = 'yz',output_width = None, output_height = None, factor = 1, rot90_k = 0 ):
    
    if isinstance(pcd, np.ndarray):
        points = pcd
    else:
        points = np.asarray(pcd.points)
    
    if output_width is None:
        VOXEL_A_SIZE = 16
        VOXEL_B_SIZE = 16
        output_width = int((a_max - a_min) / VOXEL_A_SIZE)
        output_height = int((b_max - b_min) / VOXEL_B_SIZE)
    else:
        VOXEL_A_SIZE = (a_max - a_min)/output_width
        VOXEL_B_SIZE = (b_max - b_min)/output_height

    coords = []
    for I, point in enumerate(points):
        x,y,z = point[0:3]
        #if X_MIN < x < X_MAX and cfg.Y_MIN < y < cfg.Y_MAX:
        if view == 'yz':
                a, b = int((y - a_min) / VOXEL_A_SIZE *factor), int((z - b_min) / VOXEL_B_SIZE * factor)
        elif view =='xy':
                a, b = int((x - a_min) / VOXEL_A_SIZE *factor), int((y - b_min) / VOXEL_B_SIZE * factor)
        elif view == 'xz':
                a, b = int((x - a_min) / VOXEL_A_SIZE *factor), int((z - b_min) / VOXEL_B_SIZE * factor)
        coords.append((b,a))
        # coords.append((b,output_width-a))
    return coords

def yz_to_bev_yz(centroid_yz, VOXEL_SIZE, y_offset, z_offset):
        for I, (y,z) in enumerate(centroids_yz):
            y = int((y - self.y_min) / VOXEL_SIZE + y_offset)
            z = int((z - self.z_min) / VOXEL_SIZE + z_offset)
        centroid_bev.append((y, z))
   

