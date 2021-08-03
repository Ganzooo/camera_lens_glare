CUDA_VISIBLE_DEVICES=2 python3 src/train_no_apex_refactor.py --batch_size 16 --max_epoch 100 \
--data_path /workspace/gz/camera_light_glare/patches_768_0/ --width 512 --height 512 --model_type resnet_unet \
--loss_type mse --result_dir ./result_act_768_sgd_lambda/ --work_dir ./cur/deglare_act_768_sgd_lambda/ \
--submission_dir ./submission_act_768_sgd_lambda/ \
--lr_initial 0.002 --optimizer sgd --scheduler cosine \
--resume ./cur/deglare_act_768_sgd_lambda/glare_bestmodel_27.021099090576172 --save_image_train True
