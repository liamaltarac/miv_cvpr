## Code thanks to : https://keras.io/examples/vision/visualizing_what_convnets_learn/

import tensorflow as tf
from tensorflow.keras.initializers import Initializer
from keras import backend as K
import tensorflow_probability as tfp
import math
import numpy as np
from scipy.stats import wrapcauchy
import math as m
class Init(Initializer):

    def __init__(self, rho=0.9, rho2 = 0.99, beta=2/3, sample_batch=100, rgb=False, malus=True, n=64, act_fct='relu', unipolar=False):        #n_channels = None
        #self.filters = None
        self.n_avg = None
        self.k = None

        self.rho = rho
        self.beta = beta
        
        self.filters = None

        self.input = None
        self.batch = sample_batch

        self.rho2 = rho2
        self.rgb = rgb
        self.malus = malus
        self.n = n

        self.act_fct = act_fct
        self.unipolar = unipolar
        #self.orr_samp = orrientation_sampling

    def getSymAntiSymTF(self, filter):

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

        fs1 = tf.expand_dims(a+c+g+i, 0)/4
        fs2 = tf.expand_dims(b+d+f+h,0)/4
        fs3= tf.expand_dims(e, 0)

        sym = tf.stack([tf.concat([fs1, fs2, fs1],  axis=0), 
                            tf.concat([fs2, fs3, fs2], axis=0),
                            tf.concat([fs1, fs2, fs1], axis=0)])
            
        anti = filter - sym

        return sym, anti

    def __call__(self, shape, dtype=None, **kwargs):
        n_channels = int(shape[-2])
        n_filters = int(shape[-1])
        k = shape[0]      

        print("MALUS 1:", self.malus, n_channels, self.rgb)
        if n_channels<=3:
            self.rgb = True
            malus = False
        else:
            self.rgb = False
            malus = True  #self.malus


        #t = tf.sort(tfp.distributions.Uniform(0, (n_filters//8)*2*np.pi).sample(sample_shape=(1,n_channels, n_filters)), axis=-2) + np.pi/4
        
        #t = tf.sort(tfp.distributions.Uniform(0, 2*np.pi*(n_filters//self.n)-(2*np.pi/self.n)).sample(sample_shape=(1,n_channels, n_filters)), axis=1) #+ np.pi/4
        
        
        
        #t = tf.sort(tfp.distributions.Uniform(0, 2*np.pi*(n_channels//self.n)).sample(sample_shape=(1,n_channels, n_filters)), axis=1) #+ np.pi/4
    
        print("MALUS : ", self.malus)
        if self.rgb:
            t = tf.linspace(0., 2*np.pi*(n_filters//self.n)-2*np.pi/self.n , n_filters, axis=0)
            t = tf.expand_dims(tf.transpose(tf.stack([t] * n_channels, axis=1)), axis=0) #tf.tile(filter_prev_rotation, [n_filters]) #, (1, 1, n_channels, n_filters))
            print(t)

        t -= np.pi/4
        #print(t)
        
        #print("T",t[:,:,0])
        # Use copula method to create ra2 and rs2 distributions that are c orrelated by rho2
        cov = tf.stack([1,         self.rho2,      
                        self.rho2,          1 ])
        cov = tf.cast(tf.reshape(cov, (1, 2,2)), tf.dtypes.float32)

        dist = tfp.distributions.MultivariateNormalFullCovariance(
            loc=0,
            covariance_matrix=cov,
            validate_args=False,
            allow_nan_stats=True,
        )

        sample = dist.sample(sample_shape = (1, n_channels, n_filters)) 

        x = sample[:,:,:,:,0]
        y = sample[:,:,:,:,1]# = sample[:,:,2]

        uniform =  tf.clip_by_value(tfp.distributions.Normal(0, 1).cdf(sample), K.epsilon(), 1-K.epsilon())
        
        a = 1 * self.beta  #4.5 * self.beta
        b = 1-a             #4.5 - a

        '''if self.beta == 0.5:
            a, b = 1., 1.
        if self.beta > 0.5:
            a, b = -self.beta/(self.beta-1), 1.
        if self.beta < 0.5:
            a, b = 1., (1/self.beta) - 1 '''

        #print(a, b)

        ra2 = tf.squeeze(tfp.distributions.Gamma(a, 1).quantile(uniform[:,:,:,:,0]), axis=-1)
        rs2 = tf.squeeze(tfp.distributions.Gamma(b, 1).quantile(uniform[:,:,:,:,1]), axis=-1)
        print("A, B", a, b)


        '''if self.malus:
            #print(filter_rotation.shape,filter_prev_rotation.shape, (x**2 + y**2).shape)
            filter_prev_rotation = tf.reshape(tf.repeat(tf.linspace(0., 2*np.pi- 2*np.pi/self.n ,self.n, axis=0), n_channels//self.n), (-1))#+ np.pi/4
            filter_prev_rotation = tf.reshape( tf.tile(filter_prev_rotation, [n_filters]), (1, n_channels, n_filters))
            
            #filter_rotation = tf.reshape(tf.repeat(tf.linspace(0., 2*np.pi-2*np.pi/self.n , self.n, axis=0), 1), (-1,1 )) #+ np.pi/4
            filter_rotation = tf.reshape(tf.repeat(tf.linspace(0., 2*np.pi-2*np.pi/self.n , self.n, axis=0), n_channels//self.n), (-1,1 )) 

            #filter_rotation = tf.gather(filter_rotation, tf.range(0, n_filters, delta=n_filters//n_channels)) -  np.pi/4
            #print("FPR ", filter_prev_rotation.shape , self.filters.shape, filter_rotation.shape)
            #print("FPR ", filter_prev_rotation.shape , self.filters.shape, tf.reshape(filter_rotation, (1, 1, n_channels, 1)).shape)            
            #print(tf.reshape(filter_rotation, (1, 1, n_channels, 1))-filter_prev_rotation)
            print("RA2 :", ra2.shape)
            print("RS2 :", rs2.shape)

            ra2 = ra2 * tf.math.cos((tf.reshape(filter_rotation, (1, n_channels, 1))-filter_prev_rotation))**2
            rs2 = rs2 * tf.math.cos((tf.reshape(filter_rotation, (1, n_channels, 1))-filter_prev_rotation))**2
            print("RA2 :", ra2.shape)
            print("RS2 :", rs2.shape)
            #print(filter_prev_rotation)
            #print(filter_rotation)
        '''
        #self.filters = asym_filters + sym_filters
        #ra2 = tf.sqrt(tfp.distributions.Chi2(1).sample(sample_shape=(1,n_channels, n_filters)))

        #ra2 = tf.squeeze(tfp.distributions.Gamma(.05, .1).quantile(uniform[:,:,:,:,0]), axis=-1)

        #rs2 = tfp.distributions.Gamma(.05, .1).quantile(uniform[:,:,:,:,1])

        # Use "coloring" to reshape the antisymetric distribution (add correlation in the antisymetric plane) 
        #ra2 = 1.
        x = tf.sqrt(ra2)*tf.math.cos(t)
        y = tf.sqrt(ra2)*tf.math.sin(t) 
        self.antisym_dist  = tf.stack([x, y], axis=1)[0]
        print("AD shape, ", self.antisym_dist.shape)

        if not self.rgb :

            cov = tf.stack([1,         self.rho,      
                            self.rho,          1 ])
            cov = tf.cast(tf.reshape(cov, (1, 2,2)), tf.dtypes.float32)

            #filter_rotation = tf.reshape(tf.linspace(0., (n_filters//8)*2*np.pi, n_filters, axis=0), (n_filters,1 ))+ np.pi/4
            #filter_rotation = tf.reshape(tf.linspace(0., 2*np.pi, n_filters, axis=0), (n_filters,1 ))
            filter_rotation = -(tf.reshape(tf.linspace(0., 2*np.pi*(n_filters//self.n)-2*np.pi/self.n , n_filters, axis=0), (-1,1 )) + np.pi/2)

            rot = tf.stack([tf.concat([tf.math.cos(filter_rotation),          -tf.math.sin(filter_rotation)], axis=-1),      
                            tf.concat([tf.math.sin(filter_rotation),           tf.math.cos(filter_rotation)], axis=-1)], axis=-1)
            cov = rot @ cov @ tf.linalg.matrix_transpose(rot)

            self.antisym_dist  = tf.transpose(self.antisym_dist, perm=[2, 0, 1])
            #self.antisym_dist = self._color(self.antisym_dist, cov)
            self.antisym_dist  = tf.transpose(self.antisym_dist, perm=[1, 2, 0])

        self.antisym_dist  = tf.expand_dims(self.antisym_dist, axis=0)
        x, y = self.antisym_dist[0,0,:,:], self.antisym_dist[0,1,:,:]

            #Get new coords for the kernels
        #ra2 = x**2 + y**2


        #uniform = tf.clip_by_value(tfp.distributions.Chi2(2).cdf(ra2) , K.epsilon(), 1-K.epsilon()) #Use inverse CDF method to transform ra2 into a more sparse distribution
        #print(uniform.shape, ra2.shape)
        #print(uniform)
        #ra2 = tfp.distributions.Gamma(1/(.1*(n_channels-0.01)) , 1).quantile(uniform)
        
        #print("RA2:")
        #print(ra2)
        theta = t

        ra = tf.math.sqrt(ra2)
        print("TFSHAPE ", theta.shape)

        theta = theta #+ (tf.cast(tf.random.uniform(shape = (n_channels, n_filters), maxval=2, dtype=tf.int32), dtype=tf.float32)*np.pi) #- (m.pi/4)
        if self.unipolar :
            print("TFSHAPE ", theta.shape, filter_rotation.shape)
            filter_rotation=tf.reshape(-tf.linspace(0., 2*np.pi*(n_filters//self.n)-2*np.pi/self.n , n_filters, axis=0), (-1,1 ))+(np.pi/4)
            ##print(tf.linspace(0., 2*np.pi*(n_filters//self.n)-2*np.pi/self.n , n_filters, axis=0)[11])
            print(filter_rotation[0])
            #print((tf.reshape(-1*tf.linspace(0., 2*np.pi*(n_filters//self.n)-2*np.pi/self.n , n_filters, axis=0), (-1,1 )) + np.pi/2)[0])

            print(theta[:,0])
            print(tf.less(tf.math.cos(-tf.transpose(filter_rotation)-theta), 0)[:,0])

            theta = tf.where(tf.less(tf.math.cos(tf.transpose(filter_rotation)-(-theta)), 0), theta+np.pi, theta)
        #theta = tf.expand_dims(theta, axis=0)
            
        if self.rho > 0.0 or self.rho < 0.:
            pass #theta -=  (m.pi/4)
        #print("THETA :, ", theta)
        #print("RA2 = ", ra2.shape)


        # Generate Filters
        a = -tf.math.sin(tf.math.cos(theta)- tf.math.sin(theta))
        b = tf.math.sin(tf.math.sin(theta))
        c = tf.math.sin(tf.math.cos(theta) + tf.math.sin(theta))
        d = -tf.math.sin(tf.math.cos(theta))
    
        asym_filters = tf.stack([tf.concat( [a,b,c], axis=0) , 
                        tf.concat( [d,tf.zeros([1, n_channels, n_filters]), -d], axis=0),
                        tf.concat( [-c, -b, -a], axis=0)])
        
        #Scale the filters to the correct magnitude from the  Anti-symetric distribution  
        norm = tf.sqrt(tf.reduce_sum(tf.square(asym_filters), axis=[0,1]))  
        asym_filters =  tf.math.multiply((asym_filters / norm) , ra)

        
        sym_filters = tf.ones(shape=(3,3, n_channels, n_filters))
        #print("RS2 RA2 ", rs2.shape, ra2.shape )

        v_ra2 = tfp.stats.variance(
            tf.cast(ra2, dtype=tf.float32), sample_axis=None, keepdims=True
        ) 

        v_rs2 = tfp.stats.variance(
            tf.cast(rs2, dtype=tf.float32), sample_axis=None, keepdims=True
        ) 


        #print("V rs2 :", v_rs2)
        #print("V ra2 :", v_ra2)

        #desired_e_rs2 = tf.sqrt(e_ra2 * ((1. / self.beta) -1))
        #rs = tf.squeeze(tf.sqrt(desired_e_rs2 * rs2/e_rs2))
        #print("RS2 HERE: ", rs2.shape)
        #desired_std_rs2 = tf.sqrt(v_ra2 * ((1. / self.beta) -1))
        #print("DES :", desired_std_rs2)
        rs = tf.sqrt(rs2)  #desired_std_rs2 * rs2/tf.sqrt(v_rs2))


        #print("New V rs2 :",tfp.stats.variance(rs**2, sample_axis=None))
        #print("New V ra2 :",tfp.stats.variance(ra**2, sample_axis=None))

        norm = tf.sqrt(tf.reduce_sum(tf.square(sym_filters), axis=[0,1]))  
        sym_filters =  tf.math.multiply((sym_filters / norm) , rs)
        
        dc = tf.random.uniform(shape = (n_channels, n_filters), maxval=2, dtype=tf.int32)*2 - 1

        sym_filters = sym_filters  * tf.cast(dc, dtype=tf.float32)

        self.filters  = asym_filters + sym_filters

        if malus:
            #print(filter_rotation.shape,filter_prev_rotation.shape, (x**2 + y**2).shape)
            #filter_prev_rotation = tf.reshape(tf.repeat(tf.linspace(0., 2*np.pi- 2*np.pi/self.n ,self.n, axis=0), n_channels//self.n), (-1))#+ np.pi/4
            filter_prev_rotation = tf.reshape(tf.linspace(0., 2*np.pi*(n_channels//self.n)-2*np.pi/self.n , n_channels, axis=0), (-1)) #+ np.pi/4
            print("filter_prev_rotation :", filter_prev_rotation.shape)

            filter_prev_rotation =  tf.stack([filter_prev_rotation] * n_filters) - np.pi/4 #tf.tile(filter_prev_rotation, [n_filters]) #, (1, 1, n_channels, n_filters))
            #np.savetxt('fpr.csv',filter_prev_rotation.numpy(),delimiter=',',fmt='%10.5f')

            #filter_rotation = tf.reshape(tf.repeat(tf.linspace(0., 2*np.pi-2*np.pi/self.n , self.n, axis=0), 1), (-1,1 )) #+ np.pi/4
            #filter_rotation = tf.reshape(tf.repeat(tf.linspace(0., 2*np.pi-2*np.pi/self.n , self.n, axis=0), n_channels//self.n), (-1,1 )) 
            
            filter_rotation = tf.reshape(tf.linspace(0., 2*np.pi*(n_filters//self.n)-(2*np.pi/self.n) , n_filters, axis=0), (-1,1 )) - np.pi/4
            #np.savetxt('fr.csv',filter_rotation.numpy(), delimiter=',',fmt='%10.5f')

            #filter_rotation = tf.gather(filter_rotation, tf.range(0, n_filters, delta=n_filters//n_channels)) -  np.pi/4
            print("FPR ", filter_prev_rotation.shape , self.filters.shape, filter_rotation.shape)
            print("FPR ", filter_prev_rotation.shape , self.filters.shape, (filter_rotation-filter_prev_rotation).shape)            
            #print(tf.reshape(filter_rotation, (1, 1, 1, n_filters))-filter_prev_rotation)
            #np.savetxt('malus.csv', tf.transpose(filter_rotation-filter_prev_rotation), delimiter=',',fmt='%10.5f')
            #np.savetxt('malusNoRe.csv', filter_rotation-filter_prev_rotation, delimiter=',',fmt='%10.5f')
            print("MALUS")
            print((tf.math.cos(tf.transpose(filter_rotation-filter_prev_rotation))**2))
            self.filters = self.filters * tf.math.cos(tf.transpose(filter_rotation-filter_prev_rotation))**2
        #self.filters = tf.where(tf.equal(self.filters, 0.), K.epsilon(),  self.filters)
        #print(filter_prev_rotation)
        #print(filter_rotation)'''
        
        #self.filters = asym_filters + sym_filters

        '''if self.rgb:
            self.filters = asym_filters'''

        
        #output = tf.nn.conv2d(self.input, self.filters , 1, 'VALID')
        if self.act_fct == 'relu':
            self.input = tf.keras.activations.relu(tf.random.normal(stddev=1.0, shape=[self.batch, 30,30, n_channels],))
            output = tf.keras.activations.relu(tf.nn.conv2d(self.input, self.filters , 1, 'VALID'))

        else:
            self.input = tf.random.normal(stddev=1.0, shape=[self.batch, 30,30, n_channels])
            output = tf.nn.conv2d(self.input, self.filters , 1, 'VALID')


        input_var = tfp.stats.variance(
            self.input, sample_axis=None)      

        output_var= tfp.stats.variance(
            output, sample_axis=None)
        
        
        print("VARS ", input_var, output_var, tf.math.reduce_mean(self.filters, axis=None))
        scale = tf.math.sqrt(input_var/output_var)

        self.filters = self.filters * scale

        if self.act_fct == 'relu':
            self.input = tf.keras.activations.relu(tf.random.normal(stddev=0.1, shape=[self.batch, 30,30, n_channels],))
            output = tf.keras.activations.relu(tf.nn.conv2d(self.input, self.filters , 1, 'VALID'))

        else:
            self.input = tf.random.normal(stddev=0.1, shape=[self.batch, 30,30, n_channels])
            output = tf.nn.conv2d(self.input, self.filters , 1, 'VALID')

        input_var = tfp.stats.variance(
            self.input, sample_axis=None)      

        
        #output = tf.nn.conv2d(self.input, self.filters , 1, 'VALID')
       
        output_var= tfp.stats.variance(
            output, sample_axis=None)

        print("NEW VARS ", scale.numpy()  , input_var.numpy(), output_var.numpy())

        print("F ", tfp.stats.variance(self.filters, sample_axis=None)  , tf.math.reduce_mean(self.filters, axis=None))


        return (self.filters)

    def get_config(self):  # To support serialization
        return {}

    def __str__(self):
        return f"GeometricInit3x3(p={self.rho}, b={self.beta})"
        
    '''def rgb_filters(self, shape, beta):
        n_channels = int(shape[-2])
        n_filters = int(shape[-1])
        k = shape[0]      

        #t = tf.sort(tfp.distributions.Uniform(0, (n_filters//8)*2*np.pi).sample(sample_shape=(n_filters)), axis=0) + np.pi/4
        #t = tf.sort(tfp.distributions.Uniform(0, 2*np.pi).sample(sample_shape=(n_filters)), axis=0) + np.pi/4
        t = tf.reshape(tf.linspace(0., 2*np.pi*(n_filters//self.n)-2*np.pi/self.n, n_filters, axis=0), (-1))#+ np.pi/4

        t = tf.expand_dims(tf.stack([t]*n_channels, axis=0), axis=0)
        print('THETA  : ', t.shape)

        ra2 = tfp.distributions.Uniform(low=0.99,high=1.01).sample(sample_shape=(n_channels, n_filters)) #,tf.ones(shape = (n_channels, n_filters))
        x = tf.sqrt(ra2)*tf.math.cos(t[0])
        print(t[0])
        y = tf.sqrt(ra2)*tf.math.sin(t[0]) 
        ra = tf.math.sqrt(ra2)
        theta = tf.expand_dims(tf.math.atan2(y, x), axis=0) #+ np.pi/4
        #theta = theta + (tf.cast((tf.random.uniform(shape = (1, n_channels,  n_filters), maxval=2, dtype=tf.int32)), dtype=tf.float32) * np.pi)

        a = -tf.math.sin(tf.math.cos(theta)- tf.math.sin(theta))
        b = tf.math.sin(tf.math.sin(theta))
        c = tf.math.sin(tf.math.cos(theta) + tf.math.sin(theta))
        d = -tf.math.sin(tf.math.cos(theta))
    
        asym_filters = tf.stack([tf.concat( [a,b,c], axis=0) , 
                        tf.concat( [d,tf.zeros([1, n_channels, n_filters]), -d], axis=0),
                        tf.concat( [-c, -b, -a], axis=0)])
        
        #Scale the filters to the correct magnitude from the  Anti-symetric distribution  
        norm = tf.sqrt(tf.reduce_sum(tf.square(asym_filters), axis=[0,1]))  
        asym_filters =  tf.math.multiply((asym_filters / norm) , ra)


        sym_filters = tf.ones(shape=(3,3, n_channels, n_filters))

        rs = tf.squeeze(tf.sqrt(ra2 * ((1. / self.beta) -1)))
        dc = tf.random.uniform(shape = (1, 1, n_filters), maxval=2, dtype=tf.int32)*2 - 1

        norm = tf.sqrt(tf.reduce_sum(tf.square(sym_filters), axis=[0,1]))  
        sym_filters =  tf.math.multiply((sym_filters / norm) , rs)
        
        sym_filters = sym_filters * tf.cast(dc, dtype=tf.float32)

        self.filters = asym_filters + sym_filters


        self.input = tf.nn.relu(tf.random.normal(stddev=1.0, shape=[self.batch, 30,30, n_channels],))

        input_var = tfp.stats.variance(
            self.input, sample_axis=None)
        
        #output = tf.nn.conv2d(self.input, self.filters , 1, 'VALID')
        output = tf.keras.activations.relu(tf.nn.conv2d(self.input, self.filters , 1, 'VALID'))

        output_var= tfp.stats.variance(
            output, sample_axis=None)

        print("VARS ", input_var, output_var)
        scale = tf.math.sqrt(input_var/output_var)

        self.filters = self.filters * scale
        return (self.filters)'''