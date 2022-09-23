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
# NOTE: 1. when staging, design kernel size/stride carefully
#       2. by design, please make sure models are built before called in tf.graphs to prevent retracing, build graph function is provided for such purpose

class MDDiscriminator(Model):
    
    def __init__(self, image_shape_in, num_channel, num_domains, num_out_filter=16,
                 stage_filters=(64, 128, 256), stage_kernels=(3, 3, 3),
                 stage_strides_ds=(2,2,0)):
        super().__init__()
        
        # staging params must be equal
        assert len(stage_filters) == len(stage_kernels)
        assert len(stage_filters) == len(stage_strides_ds)
        
        self.image_shape_in = image_shape_in
        self.num_channel = num_channel
        
        # ResBLK layers, downsample blk layers separated from upsample blk layers
        
        self.relus1_ds = []
        self.convs1_ds = []
        
        self.relus2_ds = []
        self.convs2_ds = []
        
        self.adds_ds = []
        self.dss = []

        for idx in range(len(stage_filters)):
            self.adds_ds.append(layers.Add())

            self.relus1_ds.append(layers.LeakyReLU())
            self.relus2_ds.append(layers.LeakyReLU())

            self.convs1_ds.append(layers.Conv2D(stage_filters[idx],(stage_kernels[idx],stage_kernels[idx]),
                                               strides=(1, 1), padding='same'))
            self.convs2_ds.append(layers.Conv2D(stage_filters[idx],(stage_kernels[idx],stage_kernels[idx]),
                                               strides=(1, 1), padding='same'))

            if stage_strides_ds[idx] == 0:
                self.dss.append(0)

            else:
                self.dss.append(layers.Conv2D(stage_filters[idx+1], (stage_kernels[idx],stage_kernels[idx]),
                                               strides=(stage_strides_ds[idx],stage_strides_ds[idx]), padding='same'))

        
        # in conv 1x1
        self.in_conv11 = layers.Conv2D(stage_filters[0], (1, 1),
                                        strides=(1, 1), padding='same',
                                        input_shape=(None, self.image_shape_in[0],
                                        self.image_shape_in[1], self.num_channel))
        
        # final downsample
        self.fd_rl1 = layers.LeakyReLU()
        self.fd_ds = layers.Conv2D(stage_filters[-1], (stage_kernels[-1],stage_kernels[-1]),
                                               strides=(2, 2), padding='same')
        self.fd_rl2 = layers.LeakyReLU()
        
        # reshape for linear act
        self.rs_out = layers.Reshape((int(self.image_shape_in[0]/(2**(len(stage_strides_ds))))*int(self.image_shape_in[1]/(2**(len(stage_strides_ds))))*stage_filters[-1],))
        
        # style code production
        self.rf_layers = []
        for i in range(num_domains):
            self.rf_layers.append(layers.Dense(num_out_filter))
            
    def call(self, img_in):
            
        # conv 1x1
        x = self.in_conv11(img_in)
        
        # downsample blks
        for idx in range(len(self.adds_ds)):
            x_res = self.convs1_ds[idx](self.relus1_ds[idx](x))
            x_res = self.convs2_ds[idx](self.relus2_ds[idx](x_res))
            x = self.adds_ds[idx]([x, x_res])
            if not self.dss[idx]==0:
                x = self.dss[idx](x)
                

        # final downsample
        x = self.fd_rl2(self.fd_ds(self.fd_rl1(x)))
        
        # reshape for linear act
        x = self.rs_out(x)
        
        rfs = []
        for rf in self.rf_layers:
            rfs.append(rf(x))
        
        return tf.transpose(tf.convert_to_tensor(rfs), [1,0,2])
    
    def build_graph(self):
        x = layers.Input(shape=(self.image_shape_in[0],
                                        self.image_shape_in[1], self.num_channel))
        
        return Model(inputs=x, outputs=self.call(x))
        
        