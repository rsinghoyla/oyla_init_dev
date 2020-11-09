import glob
import os
import sys
import random
import shutil

def chunk(xs, n):
    ys = list(xs)
    random.shuffle(ys)
    size = len(ys) // n
    leftovers= ys[size*n:]
    for c in range(n):
        if leftovers:
           extra= [ leftovers.pop() ] 
        else:
           extra= []
        yield ys[c*size:(c+1)*size] + extra

data_dir = sys.argv[1]
_d = os.path.join(data_dir,'kitti')
__d = os.path.join(_d,'2d')

if not os.path.exists(__d):
    print('no 2d files in kitti sub directory')
    exit(0)
    
rgb_files = glob.glob(os.path.join(__d,'rgb/*.png'))
depth_files = glob.glob(os.path.join(__d,'depth/*.png'))

assert len(rgb_files) == len(depth_files)
assert len(rgb_files) > 0

splits=['rg','sk','rs','sj']


os.makedirs(os.path.join(__d,'label_xml'))
# os.makedirs(os.path.join(data_dir,'annotated'))
# os.makedirs(os.path.join(os.path.join(data_dir,'annotatated'),'3d'))
# os.makedirs(os.path.join(os.path.join(data_dir,'annotatated'),'2d'))
# os.makedirs(os.path.join(os.path.join(os.path.join(data_dir,'annotatated'),'2d')),'rgb')
# os.makedirs(os.path.join(os.path.join(os.path.join(data_dir,'annotatated'),'2d')),'depth')

_rgb_files = list(chunk(rgb_files,len(splits)))
_depth_files = [[_f.replace('rgb','depth') for _f in _files] for _files in _rgb_files]

for i,s in enumerate(splits):
    os.makedirs(os.path.join(data_dir,'to_annotate_'+s))
    os.makedirs(os.path.join(os.path.join(data_dir,'to_annotate_'+s),'3d'))
    os.makedirs(os.path.join(os.path.join(data_dir,'to_annotate_'+s),'2d'))
    os.makedirs(os.path.join(os.path.join(os.path.join(data_dir,'to_annotate_'+s),'2d'),'rgb'))
    os.makedirs(os.path.join(os.path.join(os.path.join(data_dir,'to_annotate_'+s),'2d'),'depth'))
    print(os.path.join(os.path.join(os.path.join(data_dir,'to_annotate_'+s),'2d'),'depth'))
    for _f in _depth_files[i]:
        print(_f)
        shutil.copy(_f,os.path.join(os.path.join(os.path.join(data_dir,'to_annotate_'+s),'2d'),'depth'))
    for _f in _rgb_files[i]:
        shutil.copy(_f,os.path.join(os.path.join(os.path.join(data_dir,'to_annotate_'+s),'2d'),'rgb'))
