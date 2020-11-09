import numpy as np
from bokeh.plotting import output_file, show
from bokeh.tile_providers import STAMEN_TERRAIN_RETINA, OSM, get_provider
from bokeh.plotting import figure

from pyproj import Proj, transform

# create an array of RGBA data
N = 20
img = np.empty((N, N), dtype=np.uint32)
view = img.view(dtype=np.uint8).reshape((N, N, 4))
for i in range(N):
    for j in range(N):
        view[i, j, 0] = int(255 * i / N)
        view[i, j, 1] = 158
        view[i, j, 2] = int(255 * j / N)
        view[i, j, 3] = 128

# transform lng/lat to meters
from_proj = Proj(init="epsg:4326")
to_proj = Proj(init="epsg:3857")
x, y = transform(from_proj, to_proj, -122.250242, 37.457234)
tile_provider = get_provider(OSM)
fig = figure()#plot_height=100, plot_width=100) #gets zoom level
fig.add_tile(tile_provider)
r = fig.image_rgba(image=[img],
               x=[x-87],
               y=[y],
               dw=[87],
               dh=[35],
               name="image rgba",
               dh_units='data',
               dw_units='data') # dw/dh units are in meters not lat/lng
#r.glyph.anchor = "center"
output_file('test_image_rgba.html')
show(fig)
