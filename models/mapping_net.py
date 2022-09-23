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

class Mapper(Model):

    def __init__(self, noise_dim, sc_dims, num_domains):
        super().__init__()
        self.noise_dim = noise_dim
        self.sl1 = layers.Dense(256, activation="relu")
        self.sl2 = layers.Dense(256, activation="relu")
        self.domain_layers = []
        for i in range(num_domains):
            self.domain_layers.append([layers.Dense(256, activation="relu"), 
                                       layers.Dense(sc_dims)])

    def call(self, latent_code):

        x = self.sl1(latent_code)

        x = self.sl2(x)

        xs = []
        for dl in self.domain_layers:
            xs.append(dl[1](dl[0](x)))
        
        return tf.transpose(tf.convert_to_tensor(xs), [1,0,2])

    def build_graph(self):
        x = layers.Input(shape=(self.noise_dim,))
        return Model(inputs=x, outputs=self.call(x))
        
        