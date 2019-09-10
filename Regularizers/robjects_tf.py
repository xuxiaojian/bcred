from __future__ import print_function, division, absolute_import, unicode_literals

import sys
import os
import shutil
import math
import numpy as np
import logging
import tensorflow as tf

from Regularizers.nets_tf import *
from Regularizers.denoiseTV import *
from Regularizers.pybm3d import BM3D

from abc import ABC, abstractmethod
from collections import OrderedDict


############## Basis Class ##############

class RegularizerClass(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def prox(self,z,step,pin):
        pass

    @abstractmethod
    def eval(self,z,step,pin):
        pass

    def name(self):
        pass

############## Regularizer Class ##############

class NNClass(RegularizerClass):
    tau = 0;
    def __init__(self, sigSize):
        self.sigSize = sigSize

    def init(self):
        p = np.zeros(self.sigSize)
        return p

    def eval(self,x):
        return 0
    
    def prox(self, z, step, pin):
        return np.clip(z,0,np.inf), pin

    def name(self):
        return 'NN'

class ZeroClass(RegularizerClass):
    def __init__(self, sigSize):
        self.sigSize = sigSize

    def init(self):
        p = np.zeros(self.sigSize)
        return p

    def eval(self,x):
        return 0
    
    def prox(self, z, step, pin):
        return 0.0, pin

    def name(self):
        return 'Zero'

class TVClass(RegularizerClass):
    def __init__(self, sigSize, tau, sigma, bc='reflexive', bounds=np.array([-math.inf,math.inf]), maxiter=100):
        self.sigSize = sigSize
        self.tau = tau
        self.sigma = sigma
        self.bc = bc
        self.bounds = bounds
        self.maxiter = maxiter

    def init(self):
        p = np.zeros((self.sigSize[0],self.sigSize[1],2))
        return p

    def eval(self,x):
        filter1 = np.array([[0],[-1],[1]])
        filter2 = np.array([[0,-1,1]])
        dx = scipy.ndimage.filters.correlate(x,filter1,mode='wrap')
        dy = scipy.ndimage.filters.correlate(x,filter2,mode='wrap')
        r = self.sigma*np.sum(np.sum(np.sqrt(np.power(np.absolute(dx),2)+np.power(np.absolute(dy),2))))
        return r
    
    def red(self, z, step, pin, useNoise=False, extend_p=40):
        [x, pout, _, _] = denoiseTV(z, self.sigma/self.tau, pin, bc=self.bc, maxiter=self.maxiter, bounds=self.bounds)
        # print('denoiseTV ', np.amax(z-x))
        noise = noise if extend_p is None else noise[extend_p:extend_p+40,extend_p:extend_p+40]
        return self.tau*(z-x), pout

    def prox(self,z,step,pin):
        [x, pout, _, _] = denoiseTV(z, step*self.tau, pin, bc=self.bc, maxiter=self.maxiter, bounds=self.bounds)
        return x, pout

    def name(self):
        return 'TV'


class DnCNNClass(RegularizerClass):
    """
    A unet implementation
    
    :param channels: (optional) number of channels in the input image
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    :param kwargs: args passed to create_net function. 
    """
   
    def __init__(self, sigSize, tau, model_path, img_channels=1, truth_channels=1):
        tf.reset_default_graph()

        # basic variables
        self.img_channels = img_channels
        self.truth_channels = truth_channels
        self.tau = tau

        # reused variables
        self.nx = sigSize[0]
        self.ny = sigSize[1]

        # placeholders for input x and y
        self.x = tf.placeholder("float", shape=[None, None, None, self.img_channels])
        self.y = tf.placeholder("float", shape=[None, None, None, self.truth_channels])
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        # variables need to be calculated
        self.recons, self.input_shape_of_conv_layer = dncnn(self.x)   # use simple version of DnCNN

        self.amax = tf.reduce_max(self.recons)
        self.vars = self._get_vars()
        self.convolutional_operators = [v for v in self.vars if 'kernel:' in v.name]

        # load pretrained net to sess
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        session = tf.Session(config=config)
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.restore(self.sess, model_path)
   
    
    def _get_vars(self):
        lst_vars = []
        for v in tf.global_variables():
            lst_vars.append(v)
        return lst_vars

    
    def init(self):
        p = np.zeros([self.nx, self.ny])
        return p


    def red(self, s, step, pin, useNoise=False, extend_p=None, prob=1., phase=False):
        if len(s.shape) == 2:
            # reshape
            stemp = np.expand_dims(np.expand_dims(s, axis=-1),axis=0)
            xtemp = self.sess.run(self.recons, feed_dict={self.x: stemp, 
                                                    self.keep_prob: prob, 
                                                    self.phase: phase})

        elif len(s.shape) == 3:
            # reshape
            stemp = np.expand_dims(s, axis=-1)
            xtemp = self.sess.run(self.recons, feed_dict={self.x: stemp, 
                                                    self.keep_prob: prob, 
                                                    self.phase: phase})

        else:
            print('Incorrect s.shape')
            exit()

        if useNoise:
            noise = self.tau*xtemp.squeeze()
        else:
            noise = self.tau*(s - xtemp.squeeze())

        noise = noise if extend_p is None else noise[extend_p:extend_p+40,extend_p:extend_p+40]

        return noise, pin


    def prox(self, s, step, pin, prob=1., phase=False):
        if len(s.shape) == 2:
            # reshape
            s = np.expand_dims(np.expand_dims(s, axis=-1),axis=0)
            xtemp = self.sess.run(self.recons, feed_dict={self.x: s, 
                                                    self.keep_prob: prob, 
                                                    self.phase: phase})

        elif len(s.shape) == 3:
            # reshape
            s = np.expand_dims(s, axis=-1)
            xtemp = self.sess.run(self.recons, feed_dict={self.x: s, 
                                                    self.keep_prob: prob, 
                                                    self.phase: phase})

        else:
            print('Incorrect s.shape')
            exit()

        return xtemp.squeeze(), pin


    def eval(self, x):
        return 0

    
    def name(self):
        return 'DnCNN'


    def restore(self, sess, model_path):
        saver = tf.train.Saver(var_list=self.vars)
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)


    def lipschitz_upper(self):
        lipschitz_net = 1
        for i in range(len(self.convolutional_operators)):
            conv_operator = self.convolutional_operators[i]
            input_shape = self.input_shape_of_conv_layer[i]
            lipschitz_layer_ = self._compute_singular_values(conv_operator, input_shape)
            lipschitz_layer = self.sess.run(lipschitz_layer_)
            print('The estimate of the upper bound of the lipschitz constant of Conv {}: {}'.format(i, lipschitz_layer))
            lipschitz_net = lipschitz_net * lipschitz_layer
        return lipschitz_net

    def _compute_singular_values(self, conv, inp_shape):
        """ Find the singular values of the linear transformation
        corresponding to the convolution represented by conv on
        an n x n x depth input. """
        conv_tr = tf.cast(tf.transpose(conv, perm=[2, 3, 0, 1]), tf.complex64)
        conv_shape = conv.get_shape().as_list()
        a1 = int(inp_shape[0] - conv_shape[0])
        a2 = int(inp_shape[1] - conv_shape[1])
        padding = tf.constant([[0, 0], [0, 0],
                                      [0, a1],
                                      [0, a2]])
        transform_coeff = tf.fft2d(tf.pad(conv_tr, padding))
        singular_values = tf.svd(tf.transpose(transform_coeff, perm=[2, 3, 0, 1]), compute_uv=False)

        return tf.reduce_max(singular_values)

