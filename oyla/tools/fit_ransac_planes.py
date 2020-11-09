import open3d as o3d
import numpy as np
from matplotlib import cm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",type = str,required = True)
parser.add_argument("--frame_number",type = int,default = 10)
parser.add_argument("--num_planes",type = int,default = 2)
parser.add_argument("--dist_th",type = float,default = 2)
args = parser.parse_args()

filex = args.input_dir+'/3d/oyla_'+str(args.frame_number)+'.xyz'
pcd = o3d.io.read_point_cloud(filex,format='xyz')

pcd.paint_uniform_color([0.5, 0.5, 0.5])

outlier_cloud = pcd
inlier_clouds = []
n = args.num_planes
for i in range(n):
    print(outlier_cloud)
    plane_model, _inliers = outlier_cloud.segment_plane(distance_threshold=args.dist_th,
                                                 ransac_n=3,
                                                 num_iterations=100)
    print(len(_inliers))

    inlier_cloud = outlier_cloud.select_down_sample(_inliers)
    inlier_cloud.paint_uniform_color(cm.get_cmap('jet')(i*255//n)[:3])
    inlier_clouds.append(inlier_cloud)
    outlier_cloud = outlier_cloud.select_down_sample(_inliers, invert=True)
    print(outlier_cloud)

inlier_clouds.append(outlier_cloud)
o3d.visualization.draw_geometries(inlier_clouds)
