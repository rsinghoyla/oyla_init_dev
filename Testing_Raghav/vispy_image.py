import sys
from vispy import scene
from vispy import app
import numpy as np
from vispy.io import load_data_file, read_png

canvas = scene.SceneCanvas(keys='interactive')
canvas.size = 800, 600
canvas.show()

# Set up a viewbox to display the image with interactive pan/zoom
view = canvas.central_widget.add_view()

# Define a function to tranform a picture to gray
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Load the image
img_data = read_png('/Users/raghavsi/Downloads/mona_lisa_sm.png')

# Apply transformation
img_data = rgb2gray(img_data)

# Image visual
image = scene.visuals.Image(img_data, cmap='autumn', parent=view.scene)

# Set 2D camera (the camera will scale to the contents in the scene)
view.camera = scene.PanZoomCamera(aspect=1)
view.camera.set_range()
view.camera.flip = (0, 1, 0)

if __name__ == '__main__' and sys.flags.interactive == 0:
    app.run()
