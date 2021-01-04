import numpy as np
import cv2
import glob
import argparse
import os
from oyla.apps.utils import read_input_config
from oyla.mvc.utils import  transformation3
if __name__ == '__main__':
        # note that some of the UI parameters are here. missing range_m* assuming that that comes from parameters.csv
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_dir",type = str,required = True)
        parser.add_argument("--transform_types", type = str, default = "cartesian")
        parser.add_argument("--x_max",type = float, default = None)
        parser.add_argument("--x_min",type = float, default = None)
        parser.add_argument("--y_max",type = float, default = None)
        parser.add_argument("--y_min",type = float, default = None)
        
        parser.add_argument("--z_min",type=int,default=None)
        parser.add_argument("--z_max",type=int,default=None)
        args = parser.parse_args()
        depth_files = sorted(glob.glob(args.input_dir+'/dist_png/*.png'))
        x_max, x_min, y_max, y_min, z_max, z_min, range_max, range_min ,fov_x, fov_y = read_input_config(args)
        output_data_folder_name = args.input_dir+'/pcl_xyz'
        if not os.path.exists(output_data_folder_name):
            os.makedirs(output_data_folder_name)

        for df in depth_files:
            dist = cv2.imread(df,cv2.IMREAD_UNCHANGED)
            no_data_indices = np.where(dist==0)
            x, y, z,rcm = transformation3(dist,args.transform_types,fov_x, fov_y, no_data_indices)
            _a = np.c_[x,y,z]
            _a = _a.astype('single')

            with open(os.path.join(output_data_folder_name,df.split('/')[-1].replace('png','xyz')),'w') as fp:
                for __a in _a:
                    fp.write(str(__a[0])+' '+str(__a[1])+' '+str(__a[2])+'\n')
