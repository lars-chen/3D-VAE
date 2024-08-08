
# CNN-VAE
A Res-Net Style VAE for 3D Imaging data utilizing lightweight depth-wise separable convolutions. <br>


# Training Examples
<b>Notes:</b><br>
Avoid using a Bottle-Neck feature map size of less than 4x4 as all conv kernels are 3x3, if you do set --num_res_blocks to 0 to avoid adding a lot of model parameters that won't do much <br>
If you can only train with a very small batch size consider using GroupNorm instead of BatchNorm, aka set --norm_type to gn.<br>

```
python train_vae.py -mn test_run --dataset_root #path to dataset root#
```

```
python train_vae.py -mn test_run --load_checkpoint --dataset_root #path to dataset root#
```

```
python train_vae.py -mn test_run --latent_channels 128 --block_widths 1 2 4 8 --ch_multi 64 --dataset_root #path to dataset root#
```


```
python train_vae.py -mn test_run --image_size 128 --block_widths 1 2 4 4 8 --dataset_root #path to dataset root#
```

```
python train_vae.py -mn test_run --image_size 128 --deep_model  --latent_channels 64 --dataset_root #path to dataset root#
```
