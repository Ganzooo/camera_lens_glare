import os
import torch
import distutils.util

class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser): 
        #parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        ### General settings:
        parser.add_argument('--data_path', '-dp', type=str, help='Training data path',
                            default='/dataset_sub/camera_light_glare/patches_512_std/')
        parser.add_argument('--batch_size', '-bs', type=int, help='batch size',
                            default=10)
        parser.add_argument('--max_epoch', '-me', type=int, help='max epoch',
                            default=150)
        parser.add_argument('--pretrained_model', '-p', type=str, help='Pretrained model path',
                            #default='/workspace/NETWORK/ketiai_new/camera_lens_glare/checkpoints/glare_best_dua_ema32_23.pth')
                            default='/workspace/NETWORK/ketiai_new/camera_lens_glare/checkpoints/glare_dua_vdsr_33_18.pth')
                            #default=None)
        parser.add_argument('--width', type=int, help='feature map width',
                            default=512)
        parser.add_argument('--height', type=int, help='feature map height',
                            default=512)
        parser.add_argument('--resume', type=str, help='Train process resume cur/bcnn_latestmodel.pt',
                            #default='/workspace/NETWORK/camera_lens_glare/checkpoints/glare_bestmodel_25.205883026123047.pth')
                            #default= '/workspace/NETWORK/ketiai_new/camera_lens_glare/checkpoints/glare_latestmodel.pth')
                            default=None)
        parser.add_argument('--distributed', type=distutils.util.strtobool,help='Distributed training mode', default=False)
        
        ### Apex settings
        parser.add_argument('--apex', type=distutils.util.strtobool, help='Enable Apex mixed prediction True/False', default = True)
        parser.add_argument('--opt_level', type=str, help='Apex mixed prediction type', default = 'O1')
        parser.add_argument('--keep_batchnorm_fp32', type=str, default=None)
        parser.add_argument('--loss_scale', type=str, default=None)
        
        ###Log info setting
        parser.add_argument('--save_image', type=distutils.util.strtobool, help='Save test set images when best psnr found', default=True)
        parser.add_argument('--save_image_train', type=distutils.util.strtobool, help='Save train images firs 20 epoch', default=True)
        parser.add_argument('--save_image_val', type=distutils.util.strtobool, help='Save validation images', default=True)
        parser.add_argument('--save_print_interval', type=int, help='Save validation images', default=1)
        parser.add_argument('--submission_dir', type=str, help='Submission directory', default='./submission_512/')
        parser.add_argument('--result_dir', type=str, help='Result directory of save glare_latestmodel.pth', default='./result/')
        parser.add_argument('--work_dir', type=str, help='Work directory cur/bcnn', default='./cur/deglare_512_wd_e-6')
        
        ### Train settings
        parser.add_argument('--model_type', type=str, help='resnet_unet, Uformer16, resnet_unet_act_drop, resnet_unet_act_no_bn, resnet_unet_dua, resnet_unet_dua_vdsr, resnet_unet_mish, resnet_unet_dua_mish ', default='resnet_unet_dua_vdsr')
        parser.add_argument('--vdsr_depth', type=int, help='vdsr network depth', default=7)
        parser.add_argument('--loss_type', type=str, help='mse, l1, charb, loss_mask', default='mse')
        parser.add_argument('--color_domain', type=str, help='color_domian, rgb, ycbcr', default='rgb')
        parser.add_argument('--train_workers', type=int, help='train_dataloader workers', default=16)
        parser.add_argument('--eval_workers', type=int, help='eval_dataloader workers', default=8)
        parser.add_argument('--dataset', type=str, default ='glare_512', help='Dataset type: glare_512, ')
        parser.add_argument('--optimizer', type=str,  help='optimizer for training adamw, adam, sgd', default ='adam')
        parser.add_argument('--scheduler', type=str,  help='scheduler for training cosine, lambda, cosine_wr', default ='cosine')
        parser.add_argument('--lr_initial', type=float, help='initial learning rate', default=0.00002)
        parser.add_argument('--weight_decay', type=float, help='weight decay', default=0.00000001)
        parser.add_argument('--random_crop', type=distutils.util.strtobool, help='use random crop', default=False)
        parser.add_argument('--center_crop', type=distutils.util.strtobool, help='use center crop', default=False)
        parser.add_argument('--h2g_aug', type=distutils.util.strtobool, help='use h2g augmentation', default=False)
        parser.add_argument('--ema', type=distutils.util.strtobool, help='use ema', default=False)
        parser.add_argument('--wandb', type=distutils.util.strtobool, help='use wandb', default=False)
        parser.add_argument('--tv_change', type=distutils.util.strtobool, help='change train/validation', default=False)
        parser.add_argument('--freeze_network', type=distutils.util.strtobool, help='freeze network specific area', default=True)
        return parser