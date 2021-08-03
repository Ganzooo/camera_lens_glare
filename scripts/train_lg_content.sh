#CUDA_VISIBLE_DEVICES=0 python src/train_no_apex_refactor_tanh_content_loss.py --batch_size 25 --max_epoch 100 --data_path /mnt/nvme2n1TB/work/dataset/camera_light_glare/patches_512_all/ --width 512 --height 512 --model_type resnet_unet --loss_type mse --result_dir ./result_512_all/ --work_dir ./cur/deglare_512_all/ --submission_dir ./submission_512_all/ --random_crop False --lr_initial 0.002 --optimizer adam

#--pretrained_model /mnt/nvme2n1TB/work/source/camera_lens_glare/cur/deglare_with_tanh/glare_bestmodel_26.25862693786621.pth
# Baseline Content Loss
CUDA_VISIBLE_DEVICES=1 python src/train_no_apex_refactor_content_loss.py --batch_size 18 --max_epoch 100 --data_path /mnt/nvme2n1TB/work/dataset/camera_light_glare/patches_512_all/ --width 512 --height 512 --model_type resunet --loss_type mse --result_dir ./result_content_relu4/ --work_dir ./cur/deglare_content_relu4/ --submission_dir ./submission_content_relu4/ --lr_initial 0.00002 --optimizer adam --content_loss True --pretrained_model ./checkpoints/content_relu4_33.85.pth

#NestedUNet - deep vision, content loss
#CUDA_VISIBLE_DEVICES=0 python src/train_no_apex_refactor_tanh_content_loss.py --batch_size 12 --max_epoch 100 --data_path /mnt/nvme2n1TB/work/dataset/camera_light_glare/patches_512_all/ --width 512 --height 512 --model_type nest --loss_type mse --result_dir ./result_nest_content/ --work_dir ./cur/deglare_nest_content/ --submission_dir ./submission_nest_content/ --lr_initial 0.0002 --optimizer adam --deep_vision True --content_loss True --pretrained_model /mnt/nvme2n1TB/work/source/camera_lens_glare/cur/deglare_nest_content/glare_bestmodel_27.669584274291992.pth

