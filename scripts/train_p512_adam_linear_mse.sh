#!bin/bash

python train_no_apex_refactor.py --data_path=/mnt/sda/datasets/dacon_lg_aic/patch/patches_512_all \
  --optimizer=adam --scheduler=cosine --lr_initial=0.0002 --loss_type=mse
