CUDA_VISIBLE_DEVICE=0 python src/train_no_apex.py --batch_size 20 --max_epoch 50 --data_path /home/kt05/work/dataset/camera_light_glare/ --width 512 --height 512 --model_type resnet_unet 
