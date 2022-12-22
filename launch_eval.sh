#!/bin/bash
#python evaluate_stereo.py --restore_ckpt checkpoints/10000_kef-eth-split-1000.pth --dataset kefsentinel --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision
#python evaluate_stereo.py --restore_ckpt checkpoints/30000_kef-eth-car-scaled.pth --dataset kefcarla --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision
python evaluate_stereo.py --restore_ckpt checkpoints/37000_kef-eth-car-scaled-balanced.pth --dataset kefsentinel --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision


#python evaluate_stereo.py --restore_ckpt models/raftstereo-realtime.pth --dataset kefsentinel --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision
#python evaluate_stereo.py --restore_ckpt models/raftstereo-realtime.pth --dataset middlebury_H --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision
#python evaluate_stereo.py --restore_ckpt checkpoints/10000_kef-eth-split-1000.pth --dataset middlebury_H --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision
