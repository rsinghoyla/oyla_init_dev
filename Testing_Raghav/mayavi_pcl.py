import numpy as np
import mayavi.mlab
import numpy as np
fig = mayavi.mlab.figure(bgcolor=(0,0,0), size=(480, 480))
import sys
import os
assert len(sys.argv) == 2, "path to bin data"

total_num_of_files = len(os.listdir(sys.argv[1]))
pframe =  np.fromfile(os.path.join(sys.argv[1],'oyla_0.bin'),'single')

pframe = np.reshape(pframe,(-1,4))

pts = mayavi.mlab.points3d(
         pframe[:, 0],   # x
         pframe[:, 1],   # y
         pframe[:, 2],   # z
         pframe[:, 3],   # Height data used for shading
         mode="point", # How to render each point {'point', 'sphere' , 'cube' }
         colormap='jet',  # 'bone', 'copper',
         #color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
         scale_factor=1,     # scale of the points
         line_width=10,        # Scale of the line, if any
         figure=fig,
     )

# s = np.ones(len(pframe))
# pts = mayavi.mlab.quiver3d(pframe[:, 0],   # x
#                            pframe[:, 1],   # y
#                            pframe[:, 2],   # z
#                            s,
#                            s,
#                            s,
#                            scalars = pframe[:, 3],   # Height data used for shading
#                            mode="sphere", # How to render each point {'point', 'sphere' , 'cube' }
#                            colormap='jet',  # 'bone', 'copper',
#                            #color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
#                            scale_factor=0.85,     # scale of the points
#                            #line_width=10,        # Scale of the line, if any
#                            figure=fig,)
# pts.glyph.color_mode = 'color_by_scalar'
# pts.glyph.glyph_source.glyph_source.center = [0, 0, 0]

#print(plt.mlab_source.dataset.point_data.scalars)

@mayavi.mlab.animate
def anim(delay=10,ui=False):
    i = 0
    while True:
        pframe =  np.fromfile(os.path.join(sys.argv[1],'oyla_'+str(i)+'.bin'),'single')
        pframe = np.reshape(pframe,(-1,4))
        
        s = np.ones(len(pframe))
        pts.mlab_source.reset(x = pframe[:,0],y = pframe[:,1], z = pframe[:,2],scalars = pframe[:,3])#,u = s,v = s,w = s)
        mayavi.mlab.colorbar(orientation='vertical')
        print(mayavi.mlab.move(),mayavi.mlab.view(),mayavi.mlab.roll())
        #mayavi.mlab.view(focalpoint=(0,0,0))
        #print(np.max(pframe[:,3]))
        i = i+1
        if i>=total_num_of_files:
            i = 0
        
        yield

anim()
mayavi.mlab.show()

# import time
# import numpy as np
# from mayavi import mlab

# f = mlab.figure()
# V = np.random.randn(20, 20, 20)
# s = mlab.contour3d(V, contours=[0])

# @mlab.animate(delay=10)
# def anim():
#     i = 0
#     while i < 15:
#         #time.sleep(1)
#         s.mlab_source.set(scalars=np.random.randn(20, 20, 20))
#         i += 1
#         yield

# anim()
