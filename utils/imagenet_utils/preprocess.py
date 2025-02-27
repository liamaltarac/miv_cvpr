
import tensorflow as tf
from keras.applications import imagenet_utils

# Preprocessing instructions from 
# https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
def preprocess(x, label):

    new_size = tf.random.uniform([], minval=256, maxval=481, dtype=tf.float32)
    shape = tf.shape(x)

    h  =  tf.cast(shape[0], tf.float32)
    w =  tf.cast(shape[1], tf.float32)
    if h>w: 
        k = tf.cast(new_size / w, tf.float32)
        w = new_size
        h = h*k
    else: 
        k = tf.cast(new_size / h, tf.float32)
        h = new_size
        w = w*k

    h = tf.cast(h, tf.int32)
    w =  tf.cast(w, tf.int32)

    #x = tf.expand_dims(x, 0)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.resize(x, [h, w])
    x = tf.image.random_crop(x, [224,224,3])

    x = imagenet_utils.preprocess_input(x, mode="caffe") 

    return x, label

