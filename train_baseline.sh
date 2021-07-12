CUDA_VISIBLE_DEVICE=0 python src/train_no_apex.py --batch_size 100 --max_epoch 300 --data_path /home/kt05/work/dataset/camera_light_glare/ --width 256 --height 256 --model_type resnet_unet 
