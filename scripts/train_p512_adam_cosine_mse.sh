#!bin/bash

CUDA_VISIBLE_DEVICES=$1 python ../src/train_no_apex_refactor.py --data_path=/mnt/sda/datasets/dacon_lg_aic/patch/patches_512_all \
  --optimizer=adam --lr_initial=0.0002 --scheduler=cosine --loss_type=mse \
  --work_dir=./cur/train_p512_adam_cosine_mse_wd_e-6 --result_dir=./results/result_p512_adam_cosine_mse_wd_e-6 \
  --batch_size 8
