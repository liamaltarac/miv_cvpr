import sys
sys.path.append('../')
from utils.utils import *

import numpy as np
from scipy import ndimage

from skimage.filters import sobel_h
from skimage.filters import sobel_v
from scipy import stats

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.mplot3d as mp3d

from tensorflow.python.client import device_lib

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications import VGG16, ResNet50

from tensorflow.nn import depthwise_conv2d
from tensorflow.math import multiply, reduce_sum, reduce_mean,reduce_euclidean_norm, sin, cos, abs, reduce_variance
from tensorflow import stack, concat, expand_dims

import tensorflow_probability as tfp
import scienceplots
from mayavi  import mlab 

plt.style.use(['science', 'ieee'])
plt.rcParams.update({'figure.dpi': '300'})

model = ResNet50(weights=None,
                  include_top=False,
                  input_shape=(224, 224, 3))

conv_layers = []
for l in model.layers:
	if 'conv2d' in str(type(l)).lower():
		if l.kernel_size == (7, 7) or l.kernel_size == (3,3):
			conv_layers.append(l)

filters, _ = conv_layers[0].get_weights()
filters = filters #/ np.sqrt(reduce_variance(filters, axis=None))
theta = getSobelTF(filters)
print(filters.shape)
s, a = getSymAntiSymTF(filters)
a_mag = reduce_euclidean_norm(a, axis=[0,1])
s_mag = reduce_euclidean_norm(s, axis=[0,1])

mag = reduce_euclidean_norm(filters, axis=[0,1])
fig =  mlab.figure(size=(600, 643), bgcolor=(0.8980392156862745, 0.8980392156862745, 0.8980392156862745), fgcolor=(0, 0, 0))

mlab.clf()

for F in range(filters.shape[-1]):
    x =(a_mag[:,F]*np.cos((theta[:,F]))).numpy()*7
    y =( a_mag[:,F]*np.sin((theta[:,F]))).numpy()*7
    z =(s_mag[:,F]*np.sign(np.mean(s, axis=(0,1)))[:,F]).numpy()*7



    mlab.points3d(x[0], y[0], z[0], np.ones(z[0].shape), color=(1.,0.,0.), scale_factor=0.1)
    mlab.points3d(x[1], y[1], z[1], np.ones(z[0].shape), color=(0.,1.,0.), scale_factor=0.1)
    mlab.points3d(x[2], y[2], z[2], np.ones(z[0].shape), color=(0.,0.,1.), scale_factor=0.1)

mlab.plot3d(np.linspace(-10, 10, 100, endpoint=True), np.zeros(100), np.zeros(100), np.ones(100), color=(0,0,0), tube_radius=0.05)
mlab.plot3d( np.zeros(100), np.linspace(-10, 10, 100, endpoint=True),np.zeros(100), np.ones(100), color=(0,0,0), tube_radius=0.05)
mlab.plot3d(np.zeros(100), np.zeros(100),np.linspace(-10, 10, 100, endpoint=True),  np.ones(100), color=(0,0,0), tube_radius=0.05)

xx, yy = np.mgrid[-10.:10.01:0.01, -10.:10.01:0.1]
mlab.surf(xx, yy, np.zeros_like(xx), opacity=0.25, color=(0,0,1)) 
mlab.show()