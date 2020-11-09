# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/non_blocking_visualization.py

import open3d as o3d
import numpy as np
import copy
import os
import time
if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    dir_path='/Users/rsingh/Oyla Dropbox/Oyla/02 Technical/'+'10 Data/44x33/18_Jan_data_January_18_20_45_56//'
    source_raw = o3d.io.read_point_cloud(os.path.join(dir_path,'kitti')+'/3d/oyla_'+str(100).zfill(4)+'.xyz',format='xyz')
    target_raw = o3d.io.read_point_cloud(os.path.join(dir_path,'kitti')+'/3d/oyla_'+str(500).zfill(4)+'.xyz',format='xyz')
    
    
    source = source_raw.voxel_down_sample(voxel_size=0.02)
    target = target_raw.voxel_down_sample(voxel_size=0.02)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    #vis.add_geometry(target)
    
    save_image = False

    for i in range(10):
        if i%2 ==0:
            source = source_raw.voxel_down_sample(voxel_size=0.02)
        else:
            source = target_raw.voxel_down_sample(voxel_size=0.02)
        vis.update_geometry(source)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(2)
        if save_image:
            vis.capture_screen_image("temp_%04d.jpg" % i)
    vis.destroy_window()
