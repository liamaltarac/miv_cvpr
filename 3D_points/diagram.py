import sys
sys.path.append('./')
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

model = ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

filters = get_filter(model, -1)
filters = filters #/ np.sqrt(reduce_variance(filters, axis=None))
theta = getSobelTF(filters)
print(filters.shape)
s, a = getSymAntiSymTF(filters)
a_mag = reduce_euclidean_norm(a, axis=[0,1])
s_mag = reduce_euclidean_norm(s, axis=[0,1])

mag = reduce_euclidean_norm(filters, axis=[0,1])

F = 251
x =(a_mag[:,F]*np.cos((theta[:,F]))).numpy()*7
y =( a_mag[:,F]*np.sin((theta[:,F]))).numpy()*7
z =(s_mag[:,F]*np.sign(np.mean(s, axis=(0,1)))[:,F]).numpy()*7


fig =  mlab.figure(size=(600, 600), bgcolor=(0.8980392156862745, 0.8980392156862745, 0.8980392156862745), fgcolor=(0, 0, 0))
mlab.clf()

mlab.points3d(x, y, z, np.ones(z.shape), color=(0.8862745098039215,0.2901960784313726,0.2), scale_factor=0.05)
mlab.plot3d(np.linspace(-1, 1, 100, endpoint=True), np.zeros(100), np.zeros(100), np.ones(100), color=(0,0,0), tube_radius=0.01)
mlab.plot3d( np.zeros(100), np.linspace(-1, 1, 100, endpoint=True),np.zeros(100), np.ones(100), color=(0,0,0), tube_radius=0.01)
mlab.plot3d(np.zeros(100), np.zeros(100),np.linspace(-1, 1, 100, endpoint=True),  np.ones(100), color=(0,0,0), tube_radius=0.01)

xx, yy = np.mgrid[-1.:1.01:0.01, -1.:1.01:0.1]
mlab.surf(xx, yy, np.zeros_like(xx), opacity=0.25, color=(0,0,1)) 
mlab.show()