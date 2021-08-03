#!bin/bash

CUDA_VISIBLE_DEVICES=$1 python ../src/train_no_apex_refactor.py --data_path=/mnt/hdd4TB/sjkim/1_Database/4_DaconLG/patch/patches_512_all \
  --optimizer=sgd --lr_initial=0.001 --weight_decay=1e-5 --scheduler=cosine_wr --loss_type=loss_mask  \
  --work_dir=./cur/train_p512_sgd_coswr_mask_wd_e-6 --result_dir=./results/result_p512_sgd_coswr_mask \
  --batch_size=8

