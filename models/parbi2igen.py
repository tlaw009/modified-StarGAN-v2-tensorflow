import sys, os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from tensorflow.keras import Model
import tensorflow_addons as tfa
import tensorflow as tf
from IPython import display

# ResBLKs require a couple of hyperparams: (filters, kernel size, downsample/upsample scale(strides)) 
# each down sample blk will have a mirroring upsample blk, strides are passed in a forward fashion
# NOTE: 1. when staging, design kernel size/stride carefully
#       2. by design, please make sure models are built before called in tf.graphs to prevent retracing, build graph function is provided for such purpose

class Generator(Model):
    
    def __init__(self, image_shape_in, num_channel,
                 stage_filters=(64, 128, 256), stage_kernels=(3, 3, 3),
                 stage_strides_ds=(2,2,0), stage_strides_us=(2,2,0)):
        super().__init__()
        
        # staging params must be equal
        assert len(stage_filters) == len(stage_kernels)
        assert len(stage_filters) == len(stage_strides_ds)
        assert len(stage_filters) == len(stage_strides_us)
        
        self.image_shape_in = image_shape_in
        self.num_channel = num_channel
        
        # ResBLK layers, downsample blk layers separated from upsample blk layers
        self.ins1 = []
        self.relus1_ds = []
        self.convs1_ds = []
        
        self.ins2 = []
        self.relus2_ds = []
        self.convs2_ds = []
        
        self.adds_ds = []
        self.dss = []

        self.adains1 = []
        self.relus1_us = []
        self.convs1_us = []
        
        self.adains2 = []
        self.relus2_us = []
        self.convs2_us = []

        self.adds_us = []
        self.uss = []
        
        for idx in range(len(stage_filters)):
            self.adds_ds.append(layers.Add())
            self.adds_us.append(layers.Add())
            self.ins1.append(tfa.layers.InstanceNormalization(axis=-1))
            self.ins2.append(tfa.layers.InstanceNormalization(axis=-1))
            self.adains1.append(tfa.layers.InstanceNormalization(axis=-1,center=False,scale=False))
            self.adains2.append(tfa.layers.InstanceNormalization(axis=-1,center=False,scale=False))
            self.relus1_ds.append(layers.ReLU())
            self.relus2_ds.append(layers.ReLU())
            self.relus1_us.append(layers.ReLU())
            self.relus2_us.append(layers.ReLU())
            self.convs1_ds.append(layers.Conv2D(stage_filters[idx],(stage_kernels[idx],stage_kernels[idx]),
                                               strides=(1, 1), padding='same'))
            self.convs2_ds.append(layers.Conv2D(stage_filters[idx],(stage_kernels[idx],stage_kernels[idx]),
                                               strides=(1, 1), padding='same'))
            self.convs1_us.append(layers.Conv2D(stage_filters[-1-idx],(stage_kernels[-1-idx],stage_kernels[-1-idx]),
                                               strides=(1, 1), padding='same'))
            self.convs2_us.append(layers.Conv2D(stage_filters[-1-idx],(stage_kernels[-1-idx],stage_kernels[-1-idx]),
                                               strides=(1, 1), padding='same'))
            
            if stage_strides_ds[idx] == 0:
                self.dss.append(0)

            else:
                self.dss.append(layers.Conv2D(stage_filters[idx+1], (stage_kernels[idx],stage_kernels[idx]),
                                               strides=(stage_strides_ds[idx],stage_strides_ds[idx]), padding='same'))

            if stage_strides_us[idx] == 0:
                self.uss.append(0)
                
            else:
                self.uss.append(layers.Conv2DTranspose(stage_filters[-2-idx], (stage_kernels[idx],stage_kernels[idx]),
                                               strides=(stage_strides_us[idx],stage_strides_us[idx]), padding='same'))
        
        # in conv 1x1
        self.in_conv11 = layers.Conv2D(stage_filters[0], (1, 1),
                                        strides=(1, 1), padding='same',
                                        input_shape=(None, self.image_shape_in[0],
                                        self.image_shape_in[1], self.num_channel))
        
        # out conv 1x1
        self.out_conv11 = layers.Conv2D(self.num_channel, (1, 1),
                                        strides=(1, 1), padding='same', activation="tanh")
            
    def call(self, img_in, s_c):
        if img_in.shape[0] == None:
            mu_sc = tf.reshape(tf.reduce_mean(s_c, axis=-1), (1,1,1,1))
            sigma_sc = tf.reshape(tf.math.reduce_std(s_c, axis=-1), (1,1,1,1))
        else:
            mu_sc = tf.reshape(tf.reduce_mean(s_c, axis=-1), (img_in.shape[0],1,1,1))
            sigma_sc = tf.reshape(tf.math.reduce_std(s_c, axis=-1), (img_in.shape[0],1,1,1))      
            
        # conv 1x1
        x = self.in_conv11(img_in)
        
        # downsample blks
        for idx in range(len(self.adds_ds)):
            x_res = self.convs1_ds[idx](self.relus1_ds[idx](self.ins1[idx](x)))
            x_res = self.convs2_ds[idx](self.relus2_ds[idx](self.ins2[idx](x_res)))
            x = self.adds_ds[idx]([x, x_res])
            if not self.dss[idx]==0:
                x = self.dss[idx](x)
                
        # upsample blks
        for idx in range(len(self.adds_us)):
            x_res = self.convs1_us[idx](self.relus1_us[idx](tf.math.add(tf.math.multiply(self.adains1[idx](x), sigma_sc), mu_sc)))
            x_res = self.convs2_us[idx](self.relus2_us[idx](tf.math.add(tf.math.multiply(self.adains2[idx](x_res), sigma_sc), mu_sc)))
            x = self.adds_us[idx]([x, x_res])
            if not self.uss[idx]==0:
                x = self.uss[idx](x)
        
        x = self.out_conv11(x)
        
        return x
    
    def build_graph(self, sc_dims):
        x = layers.Input(shape=(self.image_shape_in[0],
                                        self.image_shape_in[1], self.num_channel))
        sc = tf.random.normal((sc_dims,))
        return Model(inputs=x, outputs=self.call(x,sc))
        