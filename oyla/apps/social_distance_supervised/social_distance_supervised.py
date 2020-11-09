import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import cv2
from cache_data import CacheData
from process_batch import ProcessBatch
import argparse
import os
import pickle
from oyla.apps.utils import read_input_config

parser = argparse.ArgumentParser()
parser.add_argument( "--flag_centroid_calculation", type=str, choices=["mean","median","mean_with_dbscan","centre_with_bfs"],default="mean",
                     help="choose between using 2d centeroid(0), 3d centeroid without DBSCAN(1), 3d centeroid with DBSCAN(2), for processing")
parser.add_argument( "--flag_nms", type = str, choices = ["nms", "soft_nms_d","soft_nms"], default = None)
parser.add_argument("--input_dir",default='./',help='give input folder location', required = True)
parser.add_argument("--output_file_prefix",default='social_distance_supervised',help='give output file name')
parser.add_argument('--score',type=float,default=None,help='a number between 0 and 1 which defines the threshold score for the model')
parser.add_argument("--batch_size", type = int, default = 4)
parser.add_argument("--distance_threshold", type = float, default = 200)
parser.add_argument("--flag_segmentation", type = bool, default = False)
parser.add_argument("--flag_vis", type = bool, default = False)
parser.add_argument("--flag_region", type = bool, default = False)
parser.add_argument("--flag_white",type=bool,default=False)
parser.add_argument("--flag_pcl",type=bool,default=False)
parser.add_argument("--flag_all_distances",type=bool,default=False)
parser.add_argument("--flag_tracking",type=bool,default=False)
parser.add_argument("--flag_upsample",type=bool,default=False)
parser.add_argument("--x_max",type = float, default = None)
parser.add_argument("--x_min",type = float, default = None)
parser.add_argument("--y_max",type = float, default = None)
parser.add_argument("--y_min",type = float, default = None)
parser.add_argument("--z_min",type=float,default=None)
parser.add_argument("--z_max",type=float,default=None)
parser.add_argument("--frame_start",type=int,default=-1)
parser.add_argument("--frame_end",type=int,default=100000)
parser.add_argument("--isometric", type = bool, default = False)

args=parser.parse_args()

x_max, x_min, y_max, y_min, z_max, z_min, range_max, range_min = read_input_config(args)
#print('XX',x_max,x_min)
#if(args.score>1):
#    args.score-=int(args.score)
#elif args.score<0:
#    args.score=(-1)*args.score
#    if args.score>1:
#        args.score-=int(args.score)

save_path = args.input_dir+'/collateral/'
if not os.path.isdir(save_path): os.mkdir(save_path)

output_file_prefix = args.output_file_prefix+'_'
if args.flag_segmentation:
    output_file_prefix += 'mask_'
else:
    output_file_prefix += 'bb_'
output_file_prefix += args.flag_centroid_calculation+'_'
if args.flag_nms is not None:
    output_file_prefix += 'nms_'+args.flag_nms+'_'
    
if args.score:
    output_file_prefix += 'thresh'+str(round(args.score*100))+'_'
if args.flag_upsample:
    output_file_prefix += 'upsample_'   
if args.flag_vis:
    output_file_prefix += 'vis_'
if args.flag_region:
    output_file_prefix += 'region_'
if args.flag_pcl:
    output_file_prefix += 'pcl_'
if args.flag_white:
    output_file_prefix += 'white_'
if args.flag_tracking:
    output_file_prefix += 'tracking_'


output_dict = {}
output_dict['pickle_out'] = {}#open(save_path+output_file_prefix+".pickle","wb")

im = cv2.imread('%srgb_%04d.png' % (args.input_dir,0))

scale_factor = 2
image_size = (scale_factor*int(im.shape[1]),scale_factor*int(im.shape[0]))
size = (2*scale_factor*int(im.shape[1]),scale_factor*int(im.shape[0]))
if args.isometric:
    size = (2*scale_factor*int(im.shape[1]),2*scale_factor*int(im.shape[0]))

#size = (2*int(im.shape[1]),int(im.shape[0])+int(1.0*im.shape[1]))
fourCC = cv2.VideoWriter_fourcc('M','J','P', 'G')
output_dict['video_out'] = cv2.VideoWriter(save_path+output_file_prefix+   ".avi", fourCC, 15.0, size, True)


process_batch = ProcessBatch(score = args.score, distance_threshold = args.distance_threshold, bev_shape = image_size,
                             flag_centroid_calculation = args.flag_centroid_calculation, flag_nms = args.flag_nms, 
                             output_dict = output_dict, flag_segmentation = args.flag_segmentation, flag_vis = args.flag_vis, 
                             y_max = y_max, z_max = z_max, x_max = x_max, y_min = y_min, z_min = z_min, x_min = x_min,
                             out_file_prefix=save_path+output_file_prefix, scale_factor = scale_factor, flag_white = args.flag_white, 
                             flag_pcl = args.flag_pcl, flag_all_distances = args.flag_all_distances, flag_tracking = args.flag_tracking, 
                             isometric = args.isometric, flag_region = args.flag_region,)
print(process_batch.min_size_test, process_batch.max_size_test)
cache_data = CacheData(folder_in = args.input_dir,batch_size =  args.batch_size, flag_upsample = args.flag_upsample, min_size = process_batch.min_size_test, 
                            max_size = process_batch.max_size_test, frame_start = args.frame_start, frame_end = args.frame_end)

thread_list = {}
#print('%s2d/rgb/oyla_%04d.png' % (args.input_dir,0))
for input_dict in cache_data.caching():
    # for thread_id in thread_list.keys():
    #     if thread_id.is_alive():
    #         print("forcing exit 1")
    #         thread_list[i]=True
    #         print('Joining')
    #         thread_id.join()
    #         print('thread killed')
    # thread_list={}
    
    process_batch.thread_funct(input_dict, lambda : thread_list[thread_id], )

    #thread_id = threading.Thread(target = process_frame.thread_funct, args =(input_dict,output_dict,cfg
    #                                                                        ,lambda : thread_list[thread_id], )) 
    #thread_list[thread_id]= False
    #thread_id.start()
    
# for thread_id in thread_list.keys():
#     if thread_id.is_alive():
#         print("forcing exit 1")
#         thread_list[i]=True
#         print('Joining')
#         thread_id.join()
#         print('thread killed')
output_dict['video_out'].release()
with open(save_path+output_file_prefix+".pickle","wb") as fp:
    pickle.dump(output_dict['pickle_out'],fp)
