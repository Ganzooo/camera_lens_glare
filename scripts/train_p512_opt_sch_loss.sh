#!bin/bash

python train_no_apex_refactor.py --data_path=/mnt/sda/datasets/dacon_lg_aic/patch/patches_512_all \
  --optimizer=sgd --scheduler=cosine_wr --lr_initial=0.001 --loss_type=loss_mask
