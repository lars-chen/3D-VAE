
# Conditional VAE for 3D Medical Imaging
A Res-Net Style CVAE for 3D Imaging data utilizing lightweight depth-wise separable convolutions. The data set was synthesized from the pre-trained diffusion model from [Pinaya et al. (2022)](https://arxiv.org/abs/2209.07162)

This repository serves as the basic architecture being used for 


## Training Examples

```
python train_vae.py -mn test_run --image_size 128 --deep_model --latent_channels 128 --block_widths 1 2 4 8 --ch_multi 64 --dataset_root #path to dataset root#
```

