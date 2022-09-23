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

from stargan_v2.m_starganv2 import StarGAN2

# Performs hyperparam and import checks
# relative
dataset = "afhq"

train_dataset_path=os.path.dirname(os.path.abspath(__file__))+"/datasets/"+dataset+"/train"
weight_path = os.path.dirname(os.path.abspath(__file__))+"/weights"
gen_staging_params = ((64, 128, 256), (3, 3, 3),
                 (2,2,0), (2,2,0))

se_staging_params = ((64, 128, 256), (3, 3, 3),
                 (2,2,0))

disc_staging_params = ((64, 128, 256), (3, 3, 3),
                 (2,2,0))

sgan1 = StarGAN2(train_dataset_path, (64,64), 3, 16, 64, batch_size=32,
				gen_staging_params=gen_staging_params,
				se_staging_params=se_staging_params,
				disc_staging_params=disc_staging_params
				)

sgan1.train(100)

sgan1.save_weights(weight_path)

