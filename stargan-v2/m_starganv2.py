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

from models.parbmdddiscrim import MDDiscriminator
from models.style_encoder import StyleEncoder
from models.parbi2igen import Generator
from models.mapping_net import Mapper

EPSILON = 1e-16

class StarGAN2:
    
    def __init__(self, dataset_path, image_shape, num_channel, noise_latent_dim, sc_dim, num_out_filter=16, disc_update_multi=5, 
                 batch_size=128, lr=3e-4, gp_lam=10.0, sty_lam=1.0, ds_lam=1.0, cyc_lam=1.0):
        assert len(image_shape) == 2
        assert image_shape[0]%16 == 0
        assert image_shape[1]%16 == 0
        
        self.image_shape = image_shape
        self.num_channel = num_channel
        self.noise_latent_dim = noise_latent_dim
        self.sc_dim = sc_dim
        self.batch_size, self.gp_lam, self.sty_lam, self.ds_lam, self.cyc_lam = batch_size, gp_lam, sty_lam, ds_lam, cyc_lam
        self.disc_update_multi = disc_update_multi
        self.num_domains = 1
        if not dataset_path==None:
            self.dataset = tf.keras.utils.image_dataset_from_directory(
                                  dataset_path,
                                  seed=123,
                                  image_size=self.image_shape,
                                  batch_size=self.batch_size)
            self.num_domains = len(self.dataset.class_names)
        else:
            print("WARNING: Dataset not loaded, Model in Generator mode")
        # NOTE: Dataset must be processed differently for different source and applications
        
        self.g = Generator(self.image_shape, self.num_channel)
        self.f = Mapper(self.noise_latent_dim, self.sc_dim, self.num_domains)
        self.e = StyleEncoder(self.image_shape, self.num_channel, self.sc_dim, self.num_domains)
        self.d = MDDiscriminator(self.image_shape, self.num_channel, self.num_domains, num_out_filter=num_out_filter)
        
        self.g_opt = tf.keras.optimizers.Adam(lr)
        self.f_opt = tf.keras.optimizers.Adam(lr)
        self.e_opt = tf.keras.optimizers.Adam(lr)
        self.d_opt = tf.keras.optimizers.Adam(lr)

    def adv_loss(self, dty_xt, dty_gs1, dty_gs2, dy_xy, dy_styrecg1):
        
        l2nrgs1 = tf.math.sqrt(tf.reduce_sum(tf.math.add(dty_xt, -dty_gs1)**2, axis = 1)+EPSILON)
        l2nrgs2 = tf.math.sqrt(tf.reduce_sum(tf.math.add(dty_xt, -dty_gs2)**2, axis = 1)+EPSILON)
        l2ngs12 = tf.math.sqrt(tf.reduce_sum(tf.math.add(dty_gs1, -dty_gs2)**2, axis = 1)+EPSILON)
        l2nrstyrec = tf.math.sqrt(tf.reduce_sum(tf.math.add(dy_xy, -dy_styrecg1)**2, axis = 1)+EPSILON)

        # adv_loss
        L_adv = l2nrgs1 + l2nrstyrec + l2nrgs2 - l2ngs12

        return tf.reduce_mean(L_adv)
    
    def gp_loss(self, xty_it, dty_xt, dty_gs2, tls):
        
        with tf.GradientTape() as t_gp:
            t_gp.watch(xty_it)
            dty_xit = tf.gather_nd(self.d(xty_it), tls)
            fty_xit = tf.math.add(tf.math.sqrt(tf.reduce_sum(tf.math.add(dty_xit, -dty_gs2)**2, axis = 1)+EPSILON),
                   -tf.math.sqrt(tf.reduce_sum(tf.math.add(dty_xit, -dty_xt)**2, axis = 1)+EPSILON))
            
        gp_grad = t_gp.gradient(fty_xit, xty_it)
        l2n_gp = tf.math.sqrt(tf.reduce_sum(gp_grad**2, axis = [1,2,3])+EPSILON)
        L_gp = (l2n_gp-1.0)**2
        
        return tf.reduce_mean(L_gp)
    
    def ds_loss(self, g_s1, g_s2):
        
        L_ds = tf.math.sqrt(tf.reduce_sum(tf.math.add(g_s1, -g_s2)**2, axis = [1,2,3])+EPSILON)

        return tf.reduce_mean(L_ds)
       
    def cyc_loss(self, imgs, styrec_g1, styrec_g2):
        
        l2nxygs1 = tf.math.sqrt(tf.reduce_sum(tf.math.add(imgs, -styrec_g1)**2, axis = [1,2,3])+EPSILON)
        l2nxygs2 = tf.math.sqrt(tf.reduce_sum(tf.math.add(imgs, -styrec_g2)**2, axis = [1,2,3])+EPSILON)
        
        L_cyc = l2nxygs1+l2nxygs2
        
        return tf.reduce_mean(L_cyc)
    
    def sty_loss(self, s_hat1, s_hat2, ety_gs1, ety_gs2):
        
        l2ns1 = tf.math.sqrt(tf.reduce_sum(tf.math.add(s_hat1, -ety_gs1)**2, axis = 1)+EPSILON)
        l2ns2 = tf.math.sqrt(tf.reduce_sum(tf.math.add(s_hat2, -ety_gs2)**2, axis = 1)+EPSILON)

        L_sty = l2ns1+l2ns2
        
        return tf.reduce_mean(L_sty)

    def apply_gradients(self, tapes, Ls):
        
        g_tape, d_tape, f_tape, e_tape = tapes
        L_gf, L_d, L_e = Ls
        
        grad_g = g_tape.gradient(L_gf, self.g.trainable_variables)
        grad_d = d_tape.gradient(L_d, self.d.trainable_variables)
        grad_f = f_tape.gradient(L_gf, self.f.trainable_variables)
        grad_e = e_tape.gradient(L_e, self.e.trainable_variables)
        
        self.g_opt.apply_gradients(zip(grad_g, self.g.trainable_variables))
        self.d_opt.apply_gradients(zip(grad_d, self.d.trainable_variables))
        self.f_opt.apply_gradients(zip(grad_f, self.f.trainable_variables))
        self.e_opt.apply_gradients(zip(grad_e, self.e.trainable_variables))
    
    @tf.function()
    def update(self, imgs, ls, timgs, tls, update_gen=True):
        
        noise_input1 = tf.random.normal((imgs.shape[0], self.noise_latent_dim))
        noise_input2 = tf.random.normal((imgs.shape[0], self.noise_latent_dim))
        
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape, tf.GradientTape() as f_tape, tf.GradientTape() as e_tape:
            s_hat1 = tf.gather_nd(self.f(noise_input1), tls)
            s_hat2 = tf.gather_nd(self.f(noise_input2), tls)
            
            g_s1 = self.g(imgs, s_hat1)
            g_s2 = self.g(imgs, s_hat2)
            
            dy_xy = tf.gather_nd(self.d(imgs), ls)
            dty_xt = tf.gather_nd(self.d(timgs), tls)

            dty_gs1 = tf.gather_nd(self.d(g_s1), tls)
            dty_gs2 = tf.gather_nd(self.d(g_s2), tls)

            epsi = tf.random.uniform([timgs.shape[0], 1, 1, 1], 0.0, 1.0)
            xty_it = tf.math.add(epsi*timgs, (1.0-epsi)*g_s1)
#             dty_xit = self.take_domain_output(self.d(xty_it), tls)
            
            ey_xy = tf.gather_nd(self.e(imgs), ls)
            styrec_g1 = self.g(g_s1, ey_xy)
            styrec_g2 = self.g(g_s2, ey_xy)
            dy_styrecg1 = tf.gather_nd(self.d(styrec_g1), ls)
            
            ety_gs1 = tf.gather_nd(self.e(g_s1), tls)
            ety_gs2 = tf.gather_nd(self.e(g_s2), tls)
            
            adv_loss = self.adv_loss(dty_xt, dty_gs1, dty_gs2, dy_xy, dy_styrecg1)
            gp_loss = self.gp_loss(xty_it, dty_xt, dty_gs2, tls)
            ds_loss = self.ds_loss(g_s1, g_s2)
            cyc_loss = self.cyc_loss(imgs, styrec_g1, styrec_g2)
            sty_loss = self.sty_loss(s_hat1, s_hat2, ety_gs1, ety_gs2)
            
            L_gf = adv_loss-self.ds_lam*ds_loss+self.cyc_lam*cyc_loss+self.sty_lam*sty_loss
            L_d = -adv_loss+self.gp_lam*gp_loss
            L_e = adv_loss+self.cyc_lam*cyc_loss+self.sty_lam*sty_loss
            
#         if update_gen:
        self.apply_gradients((g_tape, d_tape, f_tape, e_tape), (L_gf, L_d, L_e))


#         else:
#             grad_d = d_tape.gradient(d_loss, self.d.trainable_variables)
#             self.d_opt.apply_gradients(zip(grad_d, self.d.trainable_variables))

        return adv_loss, gp_loss, ds_loss, cyc_loss, sty_loss
        
    def train(self, epochs=250):
        num_training = 0
        for epo in range(epochs):
            adv_losses = []
            gp_losses = []
            ds_losses = []
            cyc_losses = []
            sty_losses = []
            for img_b, l_b in self.dataset:
                if not img_b.shape[0] == self.batch_size:
                    break
                if self.num_channel == 1 and img_b.shape[-1] == 3:
                    img_b = tf.image.rgb_to_grayscale(img_b)
                    
                l_b = tf.constant([[i, l_b[i].numpy()] for i in range(self.batch_size)])
                
                for timg_b, tl_b in self.dataset.take(1):
                    if self.num_channel == 1 and img_b.shape[-1] == 3:
                        timg_b = tf.image.rgb_to_grayscale(timg_b)
                        
                    tl_b = tf.constant([[i, tl_b[i].numpy()] for i in range(self.batch_size)])
                
                a_l, g_l, d_l, c_l, s_l = self.update(img_b, l_b, timg_b, tl_b)

                adv_losses.append(a_l.numpy())
                gp_losses.append(g_l.numpy())
                ds_losses.append(d_l.numpy())
                cyc_losses.append(c_l.numpy())
                sty_losses.append(s_l.numpy())
#                 if num_training%self.disc_update_multi == 0:     
#                 else:    
                num_training = (num_training+1)%self.disc_update_multi
                
            print("Epoch {:04d}".format(epo), "Avg. Adv Loss: ", np.mean(adv_losses), 
                  ", Avg. GP Loss: ",  np.mean(gp_losses),
                  ", Avg. DS Loss: ",  np.mean(ds_losses), 
                  ", Avg. CYC Loss: ",  np.mean(cyc_losses), 
                  ", Avg. STY Loss: ",  np.mean(sty_losses), flush=True)

            
    def save_weights(self, g_path, d_path):
        self.g.save_weights(g_path)
        print("Saved generator weights", flush=True)
        self.d.save_weights(d_path)
        print("Saved discriminator weights", flush=True)
    def load_weights(self, g_path, d_path):
        try:
            self.g.load_weights(g_path).expect_partial()
            print("Loaded generator weights", flush=True)
            self.d.load_weights(d_path).expect_partial()
            print("Loaded discriminator weights", flush=True)
        except ValueError:
            print("ERROR: Please make sure weights are saved as .ckpt", flush=True)
    
    def generate_samples(self, num_sam, path):
        sam_seed = tf.random.normal((num_sam, self.noise_latent_dim))
        sam_pics = self.g(sam_seed)
        for i in range(sam_pics.shape[0]):
            if self.num_channel == 1:
                plt.imshow(sam_pics[i,:,:,0], cmap='gray')
            else:   
                plt.imshow(tf.cast(tf.math.round(sam_pics[i,:,:,:]*127.5+127.5), tf.int32))
            plt.axis('off')
            plt.savefig(path+'/image_{:04d}.png'.format(i))
            plt.close('all')
            