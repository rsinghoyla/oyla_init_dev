import open3d as o3d
import numpy as np
from matplotlib import cm
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",type = str,required = True)
parser.add_argument("--frame_number",type = int,default = 10)
cmap = plt.get_cmap("tab20")
args = parser.parse_args()

filex = args.input_dir+'/oyla_'+str(args.frame_number).zfill(4)+'.xyz'
pcd = o3d.io.read_point_cloud(filex,format='xyz')

labels = np.array(pcd.cluster_dbscan(eps=100, min_points=10, print_progress=True))
max_label = labels.max()
print('unique labels with counts',np.unique(labels,return_counts=True))

colors = cmap(labels / (max_label if max_label > 0 else 1))
colors[labels <0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])
