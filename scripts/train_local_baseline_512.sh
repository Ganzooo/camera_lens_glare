CUDA_VISIBLE_DEVICES=0 python3 src/train_no_apex_refactor.py --batch_size 3 --max_epoch 100 \
--data_path /dataset_sub/camera_light_glare/patches_512_all/ \
--width 512 --height 512 --model_type resnet_unet --loss_type l1 \
--result_dir ./script/result_512_l1/ --work_dir ./cur/result_512_l1/ \
--submission_dir ./script/submission_result_512_l1/ --random_crop False \
--lr_initial 0.0002 --optimizer adam 
