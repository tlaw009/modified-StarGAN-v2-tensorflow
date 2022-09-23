# modified-StarGAN-v2-tensorflow

#### Implementation of StarGAN-v2 in tensorflow with modification to losses and model architecture.

Original Paper: https://arxiv.org/abs/1912.01865
Original implementation can be found here: https://github.com/clovaai/stargan-v2
Tensorflow variant can be found here: https://github.com/clovaai/stargan-v2-tensorflow

Mod log:
* Adversarial loss has been changed to Energy Distance as suggested in https://arxiv.org/abs/1705.10743 for unbiased gradient estimate.

* Replaces all average pooling layer with strided convolution downsampling and tranposed upsampling counterpart allowing models to learn their own down/upsampling

* down/upsampling are moved out of residual blks to limit distortion in information flows as suggested in https://arxiv.org/abs/1603.05027

NOTE: Effects of mods are not yet tested extensively(WIP), use at your own peril. 
