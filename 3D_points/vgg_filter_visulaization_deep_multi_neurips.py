#import cv2



# Install mayavi and vtk from https://www.lfd.uci.edu/~gohlke/pythonlibs/#vtk
# pip install "wheels package"

from msilib.schema import Error
from turtle import color
import numpy as np
from scipy import ndimage

l,
from skimage.filters import sobel_h
from skimage.filters import sobel_v

#from sa_decomp_layer import SADecompLayer

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  #disables GPU 
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

#tf.__version__
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications import VGG16
from tensorflow.math import multiply, reduce_sum, reduce_mean,reduce_euclidean_norm, sin, cos, abs
from tensorflow import stack, concat, expand_dims

from mayavi  import mlab 

#plt.rcParams['figure.figsize'] = [10,10]



RGB = ['R','G','B']

model = VGG16(weights="imagenet",
                  include_top=False,
                  input_shape=(224, 224, 3))

def get_filter(model, layer):

    conv_layers = []
    for l in model.layers:
        if 'conv2d' in str(type(l)).lower():
            conv_layers.append(l)
    layer = conv_layers[layer]

    # check for convolutional layer
    if 'conv' not in layer.name:
        raise ValueError('Layer must be a conv. layer')
    # get filter weights
    filters, biases = layer.get_weights()
    print("biases shape : ", biases.shape)
    print("filters shape : ", filters.shape)

    return (filters)


def getSobelTF(f):

    sobel_h = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], dtype=np.float32).reshape((3, 3, 1, 1) )/-4
    sobel_v = np.array([[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]], dtype=np.float32).reshape((3, 3, 1, 1))/-4    

    s_h = reduce_sum(multiply(f, sobel_h), axis=[0,1])
    s_v = reduce_sum(multiply(f, sobel_v), axis=[0,1])

    return (np.arctan2(s_h,s_v))



def getSymAntiSym(filter):

    #patches = extract_image_patches(filters, [1, k, k, 1],  [1, k, k, 1], rates = [1,1,1,1] , padding = 'VALID')
    #print(patches)
    mat_flip_x = np.fliplr(filter)

    mat_flip_y = np.flipud(filter)

    mat_flip_xy =  np.fliplr( np.flipud(filter))

    sum = filter + mat_flip_x + mat_flip_y + mat_flip_xy
    mat_sum_rot_90 = np.rot90(sum)
    
    return  (sum + mat_sum_rot_90) / 8, filter - ((sum + mat_sum_rot_90) / 8)

def getSymAntiSymTF(filter):

    #patches = extract_image_patches(filters, [1, k, k, 1],  [1, k, k, 1], rates = [1,1,1,1] , padding = 'VALID')
    #print(patches)
    a = filter[0,0,:,:]
    b = filter[0,1,:,:]
    c = filter[0,2,:,:]
    d = filter[1,0,:,:]
    e = filter[1,1,:,:]
    f = filter[1,2,:,:]
    g = filter[2,0,:,:]
    h = filter[2,1,:,:]
    i = filter[2,2,:,:]

    fs1 = expand_dims(a+c+g+i, 0)/4
    fs2 = expand_dims(b+d+f+h,0)/4
    fs3= expand_dims(e, 0)

    sym = stack([concat([fs1, fs2, fs1],  axis=0), 
                         concat([fs2, fs3, fs2], axis=0),
                         concat([fs1, fs2, fs1], axis=0)])
        
    anti = filter - sym

    return sym, anti



if __name__=="__main__":

    '''1 : [30, 47, 52] ,
    2 : [56, 70, 102, 107] ,
    3 : [86, 99, 14, 56, 61] ,
    4 : [87, 236, 133] ,
    5 : [128, 65, 64] ,'''

    layers = {     1 : [18, 30, 47, 29] ,  #1_2
                    2 : [3, 12, 32, 114] ,      #2_1
                    3 : [86, 100, 85, 113] ,    #2_2
                    4 : [87, 254, 113, 252] ,        #3_1
                    5 : [128, 65, 64, 74] ,         #3_2
                    6: [128, 152, 180],         #3_3
                    7: [511, 90, 126],  #4_1
                    8: [404, 14,218],   #4_2
                    9: [82, 332, 40],   #4_3
                    10:[410, 413, 226], #5_1
                    11: [108, 224 , 376] , #5_2
                    12: [115, 351, 13]

                    }
    for layer, filters in layers.items():
        for filter in filters:
            FILTER = filter
            LAYER = layer
            print("starting")

            fig =  mlab.figure(size=(1024, 1024), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            mlab.clf()

            filter_list = dict()
            sym_list = dict()
            anti_list = dict()

            zdata = np.array([])
            xdata = np.array([])
            ydata = np.array([])

            glyphs = dict()



            c = np.linspace(0, 255, 256)

            z = np.array([])
            x = np.array([])
            y = np.array([])


            filters = get_filter(model, LAYER)
            print("Got filters")

            num_filters = filters.shape[-1]
            num_channels = filters.shape[-2] #len(CHANNEL)

            thetas = getSobelTF(filters)
            s, a = getSymAntiSymTF(filters)

            s_mag = reduce_euclidean_norm(s, axis=[0,1])
            a_mag = reduce_euclidean_norm(a, axis=[0,1])
            mag = reduce_euclidean_norm(filters, axis=[0,1])[:, FILTER]

            dc = reduce_mean(filters, axis=[0,1])
            print(s_mag.shape)


            # Data for three-dimensional scattered points
            z = np.sign(dc[:,FILTER])*s_mag[:, FILTER]
            y = a_mag[:, FILTER]*np.sin(thetas[:, FILTER])
            x = a_mag[:, FILTER]*np.cos(thetas[:, FILTER])
            print(z.shape)
            filter_list = filters[:,:,:, FILTER]
            sym_list = s[:,:,:,FILTER]
            anit_list = a[:,:,:,FILTER]
            #print(spec(num_channels), i)
            rgba = (np.ones((x.shape[0], 4))*255).astype("int")
            #rgba[np.argmax(mag)] = [0, 0,0,255]

            print((x.shape[0], 4), np.argmax(mag), rgba[np.argmax(mag)])
            pts = mlab.pipeline.scalar_scatter(x, y, z)
            #print(glyphs.glyph)
            pts.add_attribute(rgba, 'colors') # assign the colors to each point
            pts.data.point_data.set_active_scalars('colors')
            glyphs = mlab.pipeline.glyph(pts)
            glyphs.glyph.glyph.scale_factor = .01 # set scaling for all the points
            glyphs.glyph.scale_mode = 'data_scaling_off' # make all the points same size

            glyphs.glyph.color_mode = 'color_by_scalar'

            zdata = np.append(zdata, z)
            ydata = np.append(ydata, y)
            xdata = np.append(xdata, x)

            beta = np.mean(np.array(a_mag)**2, axis=0)/(np.mean(np.array(a_mag**2+s_mag**2), axis=0))

            '''print(np.array2string(np.mean(np.array(a_mag**2+s_mag**2), axis=0), separator=', '))
            print(np.array2string(np.mean(np.array(a_mag**2), axis=0), separator=', '))
            print(np.array2string(np.mean(np.array(s_mag**2), axis=0), separator=', '))
            print(np.mean(np.array(a_mag)**2, axis=None)/(np.mean(np.array(a_mag**2 + s_mag**2), axis=None)))
            print(beta)'''

            #https://stackoverflow.com/questions/24308049/how-to-plot-proper-3d-axes-in-mayavi-like-those-found-in-matplotlib
            lensoffset = 0
            lim_x = np.max(np.abs(x))
            lim_y = np.max(np.abs(y))
            lim = np.max([lim_x, lim_y])
            xx = yy = zz = np.arange(-lim,lim, 2)
            xy = xz = yx = yz = zx = zy = np.zeros_like(xx)    
            '''mlab.plot3d(yx,yy+lensoffset,yz,line_width=0.01,tube_radius=0.00)
            mlab.plot3d(zx,zy+lensoffset,zz,line_width=0.01,tube_radius=0.00)
            mlab.plot3d(xx,xy+lensoffset,xz,line_width=0.01,tube_radius=0.00)'''
            cursor3d = mlab.points3d(0., 0., 0., mode='axes',
                                            color=(0, 0, 0),
                                            scale_factor=.5)

            glyph_points = glyphs.glyph.glyph_source.glyph_source.output.points.to_array()


            #print(xdata.shape,ydata.shape,zdata.shape )

            #print(zdata.shape, xdata.shape, ydata.shape)
            # Every object has been created, we can reenable the rendering.
            #fig.scene.disable_render = False

            # Here, we grab the points describing the individual glyph, to figure
            # out how many points are in an individual glyph.


            #points_dict[r] = (i, f, sym, anti)
            #print(r.actor.actor._vtk_obj)




            #mlab.axes(zlabel = "symmetric", nb_labels=1)


            # camera angle
            mlab.view(azimuth=45,  elevation=-70, distance=3, focalpoint=[0, 0,0])
            mlab.savefig(filename = f"neurips_bubble/Layer{layer}_Filer{filter}.png", figure=fig)

            mlab.close()
