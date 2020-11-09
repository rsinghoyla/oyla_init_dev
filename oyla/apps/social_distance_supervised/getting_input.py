import numpy as np
import cv2
import torch
import pandas as pd
from scipy.io import loadmat
from matplotlib import pyplot as plt
from PIL import Image
def get_input(folder,file_number, flag_upsample, min_size, max_size):
        #image_at_location=cv2.imread('%s/rgb_%04d.png' % (folder,file_number))
        orig_image_at_location=cv2.imread('%s/rgb_%04d.png' % (folder,file_number))
        dictionary_for_image={}
        if flag_upsample:
                h,w = orig_image_at_location.shape[0], orig_image_at_location.shape[1]
                dictionary_for_image['height'] = h #orig_image_at_location.shape[0] #360
                dictionary_for_image['width'] = w #orig_image_at_location.shape[1] #588
                pil_image = Image.fromarray(orig_image_at_location)

                # Resize shortest edge #
                scale = min_size * 1.0 / min(h, w)
                if h < w:
                    newh, neww = min_size, scale * w
                else:
                    newh, neww = scale * h, min_size
                if max(newh, neww) > max_size:
                    scale = max_size * 1.0 / max(newh, neww)
                    newh = newh * scale
                    neww = neww * scale
                neww = int(neww + 0.5)
                newh = int(newh + 0.5)
                pil_image = pil_image.resize((neww, newh), Image.BILINEAR) #pil_image.resize((1306, 800), Image.BILINEAR)
                image_at_location = np.asarray(pil_image)
                image_at_location = torch.as_tensor(image_at_location.astype("float32").transpose(2, 0, 1))
                dictionary_for_image['image'] = image_at_location
        else:
                dictionary_for_image['image']=torch.from_numpy(np.transpose(orig_image_at_location,(2,0,1)))
        

        # dictionary_for_image['image']=torch.from_numpy(np.transpose(image_at_location,(2,0,1)))
        
        #pcd_array=np.asarray(pd.read_table('%s3d/oyla_%04d.xyz' % (folder,file_number), sep=' ',names=['x', 'y', 'z']))
        pcd_array=np.asarray(pd.read_table('%s/pcl_%04d.xyz' % (folder,file_number), sep=' ',names=['x', 'y', 'z']))
        
        #depth = loadmat('%s2d/depth/oyla_%04d.mat' % (folder,file_number))
        depth = loadmat('%s/depth_%04d.mat' % (folder,file_number))['dist']
        #IJ=np.where(depth['dist']!=0)
        dist_img = plt.imread('%s/depth_%04d.png' % (folder,file_number))
        dist_img = np.uint8(dist_img[:,:,:3]*255)
        
        return orig_image_at_location, dictionary_for_image, depth ,pcd_array, dist_img
