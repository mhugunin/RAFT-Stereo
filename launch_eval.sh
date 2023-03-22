#!/bin/bash
python evaluate_stereo.py --restore_ckpt checkpoints/16500_kef-ir3.pth --dataset kefsentinel_dense_ir --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision
#python evaluate_stereo.py --restore_ckpt models/raftstereo-realtime.pth --dataset kefseek --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision

#python evaluate_stereo.py --restore_ckpt checkpoints/10000_kef-moredense.pth --dataset kefsentinel_dense_ir --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision

#python evaluate_stereo.py --restore_ckpt models/raftstereo-realtime.pth --dataset kefsentinel_dense_ir --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision


#python evaluate_stereo.py --restore_ckpt checkpoints/30000_kef-eth-car-scaled.pth --dataset kefcarla --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision

#run3, had carla old/bad data but good class balancing vs. the sentinel small sample
#python evaluate_stereo.py --restore_ckpt checkpoints/37000_kef-eth-car-scaled-balanced.pth --dataset kefsentinel --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision

#run4, better carla data, did not reach overfit during train duration 50k
#python evaluate_stereo.py --restore_ckpt checkpoints/50000_kef-eth-carfix.pth --dataset kefsentinel --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision

#run6, kefsolo, is just trained on sentinel with no carla pretrain. worse, and for comp purposes only

#run-small-dense-combo, the first of the good performing ones with a little dense stereo data
#python evaluate_stereo.py --restore_ckpt checkpoints/15000_kef-moredense.pth --dataset kefcarla --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision




#python evaluate_stereo.py --restore_ckpt models/raftstereo-realtime.pth --dataset kefsentinel --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision
#python evaluate_stereo.py --restore_ckpt models/raftstereo-realtime.pth --dataset middlebury_H --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision
