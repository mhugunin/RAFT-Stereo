#!/bin/bash
#python train_stereo.py --name kef-retrain --num_steps 54 --image_size 512 640 --restore_ckpt models/raftstereo-realtime.pth --train_datasets KEFSentinel --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision

#python3 train_stereo.py --train_datasets ETH3D --num_steps 4000 --image_size 384 640 --restore_ckpt models/raftstereo-realtime.pth --shared_backbone --batch_size 2 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 0 --corr_implementation reg_cuda --mixed_precision

#python train_stereo.py --name kef-eth-val --num_steps 20000 --image_size 400 400 --restore_ckpt models/raftstereo-realtime.pth --train_datasets KEFSentinel ETH3D --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision --spatial_scale 0.0 0.4 --do_flip h --do_flip v

python train_stereo.py --name kef-eth-car-scaled-balanced --num_steps 50000 --image_size 400 400 --restore_ckpt models/raftstereo-realtime.pth --train_datasets KEFSentinel ETH3D KEFCarla --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision --spatial_scale 0.0 0.4 --do_flip h --do_flip v --batch_size 10
