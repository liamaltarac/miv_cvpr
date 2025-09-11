

import sys
sys.path.append('../')

import numpy as np
from scipy import ndimage

from skimage.filters import sobel_h
from skimage.filters import sobel_v
from scipy import stats


import os
import matplotlib
matplotlib.use("Agg")                 # belt-and-suspenders
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


import scienceplots
from tensorflow.python.client import device_lib

#plt.rcParams['figure.figsize'] = [10,10]

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications import VGG16

from tensorflow.nn import depthwise_conv2d
from tensorflow.math import multiply, reduce_sum, reduce_mean,reduce_euclidean_norm, sin, cos, abs
from tensorflow import stack, concat, expand_dims

import tensorflow_probability as tfp

from utils.utils import *
import cv2

from scipy import ndimage, fft
from io import BytesIO

plt.style.use(['science', 'ieee'])
plt.rcParams.update({'figure.dpi': '100'})





#Params

ks = [2, 3, 5]   # kernel size

cs = [0.5, 1, 2, 3]  #max speed
beta2s = [1, 0, 0.25] #[0, 0.25, 0.75, 1]  
activations = [tf.nn.relu]
timestamps =  50
experiment_name = "unipolar_circle"
box_dims = [20, 16]
step =  0.01 # Plot axis step





# Single pixel input


import matplotlib.patches as mpatches

fig = plt.figure()
gs = fig.add_gridspec(1,1, wspace=0.04)

ax = fig.add_subplot(gs[0])


ax.plot(np.arange(0, step+1, step),np.arange(0, step+1, step), label="Theoretical (Lorentz)")
ax.set_xlabel(r"$\beta^2 = \frac{||f_a||^2 }{||f||^2}$")
ax.set_ylabel(r"$\beta^2 = \frac{v^2}{c^2}$")




for i, k in enumerate(ks):
    measured_beta = []
    print("K :", k)
    for beta2 in np.arange(0, 1+step, step):

        print("BETA2", beta2)

        img = np.zeros((215,215)) # cv2.imread('input4.png', 0)/255. 
        mid = img.shape[0]//2
        img[mid, mid] = 1.
        print(img.shape)



        filters = np.zeros((k,k,1,1))
        x = tf.cast(tf.repeat(tf.expand_dims([img], axis=-1) , repeats = filters.shape[-2], axis=-1), dtype=tf.float32) 


        t = np.zeros((k,k))
        
        t[0, 1] = np.sqrt(beta2)
        t[0, 0] = np.sqrt(1-beta2)
        filters = np.reshape(fft.idctn(t, norm='ortho'), (k,k,1,1)) 
        #filters /= np.sum(np.abs(filters))
        
        if k==2:

            filters = np.zeros((3,3,1,1))

            t = np.zeros((2,2))
            t[0,1] = np.sqrt(beta2)
            t[0, 0] = np.sqrt(1-beta2)






        for n in range(timestamps+1):
            x = x/np.std(x)
            vals = x[0, x.shape[1]//2, :, :]
            vals = vals/np.sum(vals)

            pos = np.expand_dims(np.linspace(-(x.shape[1]//2), x.shape[1]//2, x.shape[1]),-1)
            mean = tf.reduce_sum(pos*vals)
            var = tf.reduce_sum(((pos-mean)**2) * vals)
            std = np.sqrt(var)
            print(mean, np.sqrt(var), mid)
            

            if k==2:
                filters = np.zeros((3,3,1,1))

                if n%2 == 0:
                    filters[1:,0:2 ,0,0] = fft.idctn(t, norm="ortho")
                else:
                    filters[0:2,1: ,0,0] = fft.idctn(t, norm="ortho")
                print(filters[:,:,0,0])




            
            #else:
            #    w = tf.transpose(w, perm=(1,0,2,3))
            
            w =tf.cast(filters, dtype=tf.float32)# tf.expand_dims(filters, -1), dtype=tf.float32)


            x = tf.nn.relu( tf.nn.conv2d(x, w , strides=(1,1), 
                                    padding='SAME') )
        v = (mean)/(n)
        print( mean, mid , v, n, cs[i] , beta2)

        measured_beta.append((v/cs[i])**2)


    #ax.imshow(x[0,:,:,0])

    ax.plot(np.arange(0, step+1, step), measured_beta, label=str(k)+r"$\times$"+str(k) + r"$ \ (c = $" + str(cs[i]) + r"$)$")
    #ax.set_xlabel(r"$\beta^2 = \frac{||f_a||^2 }{||f||^2}$")
    #ax.set_ylabel(r"$\beta^2 = \frac{v^2}{c^2}$")



plt.legend()

fig.savefig(f"distance1.pdf", format="pdf", dpi=fig.dpi, bbox_inches="tight")
