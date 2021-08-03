#!/usr/bin/env python3
# coding: utf-8

import argparse
import os.path as osp
from re import I
import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
from datetime import datetime
import random
from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from dataset.dataloader import load_dataset
from models.resnet50_unet_activation import UNetWithResnet50Encoder
from models.resnet50_unet_activation_DUA import UNetWithResnet50EncoderDUA
from models.resnet50_unet_activation_bilinear import UNetWithResnet50EncoderBi
from models.resnet50_unet_activation_drop import UNetWithResnet50Encoder_act_drop
from models.resnet50_unet_activation_no_bn import UNetWithResnet50Encoder_act_no_bn
from models.resnet50_unet_activation_mish import UNetWithResnet50EncoderMish
from models.resnet50_unet_activation_DUA_mish import UNetWithResnet50EncoderDUA_mish
from models.resnet50_unet_activation_vdsr import UNetWithResnet50Encoder_vdsr
from models.resnet50_unet_activation_DUA_vdsr import UNetWithResnet50EncoderDUA_vdsr
from models.MIRNet_model import MIRNet
from models.Uformer import Uformer

from utils.utils import get_logger
from icecream import ic
from skimage import img_as_ubyte
import cv2
from utils.image_utils import save_img, batch_PSNR
import options

import apex
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers

from losses import CharbonnierLoss, CharbEdgeLoss, ContentLossWithMask, H2GLossWithMask, LossWithMask

from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from dataset.dataset_utils import MixUp_AUG

from torch_ema import ExponentialMovingAverage
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from adamp import AdamP
from adabelief_pytorch import AdaBelief
from torchtools.optim import RangerLars

#Add wanddb
import wandb
os.environ["WANDB_API_KEY"] = "0b0a03cb580e75ef44b4dff7f6f16ce9cfa8a290"
#os.environ["WANDB_MODE"] = "dryrun"

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Trainer(object):
    def __init__(self, args, logger, writer):
        self.data_path=args.data_path,
        self.batch_size=args.batch_size,
        self.max_epoch=args.max_epoch,
        self.pretrained_weight=args.pretrained_model,
        self.width=args.width,
        self.height=args.height,
        self.resume_train = args.resume,
        self.work_dir = args.work_dir,
        
        self.train_dataloader, self.val_dataloader, self.test_dataloader = load_dataset(args.data_path, args.batch_size, 
                    distributed=False, center_crop= args.center_crop, random_crop=args.random_crop, h2g_aug=args.h2g_aug, resize_size=(args.width,args.height),
                    model_type=args.model_type, color_domain=args.color_domain, tv_change=args.tv_change)
                    
        self.max_epoch = args.max_epoch
        self.time_now = datetime.now().strftime('%Y%m%d_%H%M')
        self.width = args.width
        self.height = args.height
        self.args = args
        
        self.best_psnr = 0
        self.best_test = False

        self.save_model_interval = 1
        self.loss_print_interval = 1
        self.start_epo = 0
        #self.mixup = MixUp_AUG()

        self.logdir = osp.join("./", args.work_dir)
        self.logger = get_logger(self.logdir)
        self.logger.info("Let the train begin...")
        self.test_feature_paths = list(sorted(Path(args.data_path).glob("test_input_img/*.png")))

        self.logger = logger
        self.logger.info("Let the train begin...")
        self.writer = writer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ######### Get Model ###########
        if args.model_type == "MIRNet":
            self.model = MIRNet().to(self.device)
            ic('MIRNet')
        elif args.model_type == "Uformer16":
            self.model = Uformer(img_size=self.width,embed_dim=16,win_size=8,token_embed='linear',token_mlp='leff').to(self.device)
            ic('Uformer16')
        elif args.model_type == "Uformer32":
            ic('Uformer32')
            self.model = Uformer(img_size=self.width,embed_dim=32,win_size=8,token_embed='linear',token_mlp='leff').to(self.device)
        elif args.model_type == "resnet_unet_act_drop":
            ic('ResNet50_Unet Activation Sigmoid and Drop')
            self.model = UNetWithResnet50Encoder_act_drop().to(self.device)
        elif args.model_type == "resnet_unet_act_no_bn":
            self.model = UNetWithResnet50Encoder_act_no_bn().to(self.device)
            ic('ResNet_Unet No Bn Act')
        elif args.model_type == "resnet_unet_bilinear":
            self.model = UNetWithResnet50EncoderBi().to(self.device)
            ic('ResNet_Unet_Bilinear model')
        elif args.model_type == "resnet_unet_dua":
            self.model = UNetWithResnet50EncoderDUA().to(self.device)
            ic('ResNet_Unet_DUA model')
        elif args.model_type == "resnet_unet_mish":
            self.model = UNetWithResnet50EncoderMish().to(self.device)
            ic('ResNet_Unet_Mish')
        elif args.model_type == "resnet_unet_dua_mish":
            self.model = UNetWithResnet50EncoderDUA_mish().to(self.device)
            ic('ResNet_Unet_DUA_Mish')
        elif args.model_type == "resnet_unet_dua_vdsr":
            self.model = UNetWithResnet50EncoderDUA_vdsr(args.vdsr_depth).to(self.device)
            ic('ResNet_Unet_DUA_VDSR')
        else:
            self.model = UNetWithResnet50Encoder().to(self.device)
            ic('ResNet_Unet')
        ic('Total parameter of model', sum(p.numel() for p in self.model.parameters()))

        ######### Loss ###########
        if self.args.loss_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        elif self.args.loss_type == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='mean').to(self.device)
        elif self.args.loss_type == 'charb':
            self.criterion = CharbonnierLoss().to(self.device)
        elif self.args.loss_type == 'charb_edge':
            self.criterion = CharbEdgeLoss().to(self.device)
        elif self.args.loss_type == 'loss_mask':
            self.criterion = LossWithMask(loss_type='mse').to(self.device)
        elif self.args.loss_type == 'content_loss_mask':
            self.criterion = ContentLossWithMask(loss_type='mse').to(self.device)
        elif self.args.loss_type == 'h2g_mse':
            self.criterion = H2GLossWithMask(loss_type='mse').to(self.device)
        elif self.args.loss_type == 'h2g_charb_edge':
            self.criterion = H2GLossWithMask(loss_type='charb_edge').to(self.device)
        else:
            self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)

        ######### Initialize Optimizer ###########
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr_initial, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        elif args.optimizer == 'adamp':
            self.optimizer = AdamP(self.model.parameters(), lr=args.lr_initial, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        elif args.optimizer == 'adabelief':
            self.optimizer = AdaBelief(self.model.parameters(), lr=args.lr_initial, eps=1e-16, betas=(0.9, 0.999), weight_decouple = True, rectify = False)
        elif args.optimizer == 'ranger':
            self.optimizer = RangerLars(self.model.parameters(), lr=args.lr_initial)
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr_initial, momentum=0.9, weight_decay=args.weight_decay)

        ######### Initialize APEX Mixed Prediction ###########
        if args.apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=args.opt_level)

        ######### Load Pretrained model or Resume train ###########
        if args.pretrained_model is not None:
            ic("Pretrained weight load")
            checkpoint = torch.load(args.pretrained_model)
            self.model.load_state_dict(checkpoint["model_state"])
            try:
                self.model.load_state_dict(checkpoint["model_state"])
            except:
                if args.model_type == "resnet_unet_dua_vdsr":
                    self.model.resnet_unet_dua.load_state_dict(checkpoint["model_state"])
                    # pretrained_A = args.pretrained_model[:-4] + '_A.pth'
                    # checkpointA = torch.load(pretrained_A)
                    # self.model.resnet_unet_dua.load_state_dict(checkpointA["model_state"])
                    # self.model.vdsr.load_state_dict(checkpoint["model_state"])
                else:
                    state_dict = checkpoint["state_dict"]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    self.model.load_state_dict(new_state_dict)

        if args.resume is not None:
            ic("Loading model and optimizer from checkpoint ")
            checkpoint = torch.load(args.resume)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            if args.apex:
               amp.load_state_dict(checkpoint["amp"]) 
            self.start_epo = checkpoint["epoch"]
            self.logger.info("Loaded checkpoint '{}' (epoch {})".format(args.resume, self.start_epo))
            ic(self.start_epo)
        else:
            self.logger.info("No checkpoint found at '{}'".format(args.resume))

        ######### Scheduler ###########
        warmup = False
        if warmup:
           warmup_epochs = self.start_epo
           scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch-warmup_epochs, eta_min=1e-6)
           self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
           self.scheduler.step()
        else:
            if args.scheduler == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch, eta_min=1e-6)
            elif args.scheduler == 'cosine_wr':
                self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=args.max_epoch // 3, 
                                                               cycle_mult=1.0, max_lr=args.lr_initial, min_lr=0.000001, 
                                                               warmup_steps=self.max_epoch//12, gamma=0.5)
            elif args.scheduler == 'cosine_wu':
                self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=args.max_epoch, 
                                                               cycle_mult=1.0, max_lr=args.lr_initial, min_lr=0.000001, 
                                                               warmup_steps=3, gamma=1.0)
            elif args.scheduler == 'lambda':
                self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95 ** self.max_epoch, last_epoch=-1)
            else:
                self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=10, gamma=0.1)

            # valid if resume
            for i in range(1, self.start_epo):
                self.scheduler.step()

        ic("==> Training with learning rate:", self.optimizer.param_groups[0]['lr'])
        #ic("==> Training with learning rate:", self.scheduler.get_last_lr()[0])

        ######### Initialize EMA and WandB ###########
        if args.ema:
            self.ema_model = ExponentialMovingAverage(self.model.parameters(), decay=0.995)

        if args.wandb:
            # 1. Start a W&B run
            wandb.init(project='deglare', entity='gnzrg25',reinit=True, config={"architecture":args.model_type, "dataset": args.data_path,
                    #"scheduler":args.scheduler, "lr_init":self.scheduler.get_last_lr()[0],
                    "scheduler":args.scheduler, "lr_init":self.optimizer.param_groups[0]['lr'],
                    "optim":args.optimizer, "loss":args.loss_type,
                    "batch_size":args.batch_size, "max_epoch":args.max_epoch,
                    "weight_decay":args.weight_decay}) 

            work_dir = args.work_dir
            if work_dir[-1] == '/':
                work_dir = work_dir[:-1]
            path = os.path.basename(work_dir)
            run_name = path #args.work_dir[6:]
            wandb.run.name = run_name
            wandb.run.save()
            # 2. Save model inputs and hyperparameters
            self.config = wandb.config
            #self.config.update(args)

    def step_train(self, mode):
        lr = self.optimizer.param_groups[0]['lr']
        ic('Start {} -> epoch: {}, lr: {}'.format(mode,self.epo, lr))
        self.model.train()

        self.best_test = False
        loss_sum = 0
        iter = 0

        psnr_rgb = []
        if self.args.freeze_network:
            for param in self.model.resnet_unet_dua.parameters():
                param.requires_grad = False

        tq = tqdm.tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        tq.set_description('Epoch {}, lr {}'.format(self.epo, lr))
        for index, (img, gt, fname) in tq:
            iter += 1

            img = img.to(self.device)
            gt = gt.to(self.device)



            ### Predict image ###
            pred = self.model(img)
            pred = torch.clamp(pred, 0, 1) # to prevent overflow if not use sigmoid function

            if 'mask' in self.args.loss_type:
                loss = self.criterion(pred, gt, img)
            else:
                loss = self.criterion(pred, gt)

            self.optimizer.zero_grad()
            if self.args.apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            # Update the moving average with the new parameters from the last optimizer step
            if self.args.ema:
                self.ema_model.update()

            loss_sum += loss.item()

            # PSNR
            psnr_rgb.append(batch_PSNR(pred.detach(), gt, 1.))

            # Dump train images 
            if self.args.save_image_train and np.mod(self.epo, self.args.save_print_interval) == 0:
                if iter < 20:
                    gt = gt.permute(0, 2, 3, 1).cpu().detach().numpy()
                    img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
                    pred = pred.permute(0, 2, 3, 1).cpu().detach().numpy()
                    
                    if img.shape[0] > 4:
                        temp0 = np.concatenate((img[0]*255, pred[0]*255, gt[0]*255),axis=1)  
                        temp1 = np.concatenate((img[1]*255, pred[1]*255, gt[1]*255),axis=1)  
                        temp2 = np.concatenate((img[2]*255, pred[2]*255, gt[2]*255),axis=1)  
                        temp3 = np.concatenate((img[3]*255, pred[3]*255, gt[3]*255),axis=1)  
                        temp = np.concatenate((temp0, temp1, temp2, temp3),axis=0)
                        save_img(osp.join(self.args.result_dir + '/train/' + str(self.epo) + '/batches_'+ str(iter) + '.jpg'),temp.astype(np.uint8), color_domain=self.args.color_domain)
                    else:
                        temp0 = np.concatenate((img[0]*255, pred[0]*255, gt[0]*255),axis=1)
                        save_img(osp.join(self.args.result_dir + '/train/' + str(self.epo) + '/batches_'+ str(iter) + '.jpg'),temp0.astype(np.uint8), color_domain=self.args.color_domain)

            tq.set_postfix(loss='{0:0.4f}, PSNR={1:0.4f}'.format(loss_sum / iter, sum(psnr_rgb)/len(psnr_rgb)))
        tq.close()

        self.scheduler.step()

        ### Tensorboard ###
        avg_loss = loss_sum / iter
        avg_psnr = sum(psnr_rgb)/len(psnr_rgb) #10*np.log10(1/avg_loss)
        self.writer.add_scalar("Loss/train", avg_loss, self.epo)
        self.writer.add_scalar("PSNR/train", avg_psnr, self.epo)
        self.writer.add_scalar("Scheduler/lr_value", float(lr), self.epo)
        
        ### Save latest model ###
        if np.mod(self.epo, self.save_model_interval) == 0:
            if self.args.apex:
                _state = {
                    "epoch": self.epo + 1,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "amp": amp.state_dict()
                } 
            else:
                _state = {
                    "epoch": self.epo + 1,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                } 
            _save_path_latest = osp.join(self.logdir, '{}_latestmodel.pth'.format('glare'))
            torch.save(_state,_save_path_latest)
        
        ### Print result of loss and scheduler###
        format_str = "epoch: {}\n avg loss: {:3f}, scheduler: {}"
        print_str = format_str.format(int(self.epo), float(avg_loss), float(lr))
        ic(print_str)
        self.logger.info(print_str)
        if args.wandb:
            wandb.log({"train/avg loss": avg_loss})
            wandb.log({"train/PSNR": avg_psnr})
            wandb.log({"train/lr": float(lr)})


    def step_val(self, mode):
        self.model.eval()
        
        ic('Start {} -> epoch: {}'.format(mode,self.epo))
        self.best_test = False
        psnr_val_rgb = []
        psnr_val_rgb_ema = []
        loss_sum = 0
        iter = 0

        tq = tqdm.tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader))
        tq.set_description('Validation')
        for index, (img, gt, fname) in tq:
            iter += 1
            img = img.to(self.device)
            gt = gt.to(self.device)

            with torch.no_grad():
                ###Pred image ###
                pred = self.model(img)
                pred = torch.clamp(pred,0,1) 
                psnr_val_rgb.append(batch_PSNR(pred, gt, 1.))

                #For tensorboard
                if 'mask' in self.args.loss_type:
                    loss = self.criterion(pred, gt, img)
                else:
                    loss = self.criterion(pred, gt)
                loss_sum += loss.item()

                if self.args.ema:
                    with self.ema_model.average_parameters():
                        pred_ema = self.model(img)
                        pred_ema = torch.clamp(pred_ema,0,1) 
                        psnr_val_rgb_ema.append(batch_PSNR(pred_ema, gt, 1.))

            if self.args.save_image_val and np.mod(self.epo, self.args.save_print_interval) == 0:
                gt = gt.permute(0, 2, 3, 1).cpu().detach().numpy()
                img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
                pred = pred.permute(0, 2, 3, 1).cpu().detach().numpy()
                
                for batch in range(img.shape[0]):
                    temp = np.concatenate((img[batch]*255.0, pred[batch]*255.0, gt[batch]*255.0),axis=1)
                    save_img(osp.join(self.args.result_dir + '/val/'+ str(index) + fname[batch][:-4] +'.jpg'),temp.astype(np.uint8), color_domain=self.args.color_domain)

            tq.set_postfix(loss='{0:0.4f}, PSNR={1:0.4f}, emaPSNR={2:0.4f}'.format(loss_sum/iter, sum(psnr_val_rgb)/len(psnr_val_rgb), 
                           sum(psnr_val_rgb_ema)/len(psnr_val_rgb_ema) if self.args.ema else 0))
        tq.close()

        psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
        psnr_val_rgb_ema = sum(psnr_val_rgb_ema)/len(psnr_val_rgb_ema) if self.args.ema else 0

        ### Tensorboard ###
        self.writer.add_scalar("Loss/val", loss_sum / iter, self.epo)
        self.writer.add_scalar("PSNR/val", psnr_val_rgb, self.epo)
        self.writer.add_scalar("PSNR/val_ema", psnr_val_rgb_ema, self.epo)
        if args.wandb:
            wandb.log({"val/avg loss": loss_sum / iter})
            wandb.log({"val/PSNR": psnr_val_rgb})
            wandb.log({"val/PSNR ema": psnr_val_rgb_ema})

        ### Save best model ###
        if psnr_val_rgb > self.best_psnr:
            self.best_psnr = psnr_val_rgb
            self.best_test = True
            _state = {
                "epoch": index + 1,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            } 

            _save_path = osp.join(self.logdir, '{}_bestmodel_{}.pth'.format('glare',str(float(psnr_val_rgb))))
            directory = os.path.dirname(_save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(_state, _save_path)
        
        ### Save EMA best model ###
        if self.args.ema and psnr_val_rgb_ema > psnr_val_rgb and psnr_val_rgb >= self.best_psnr:            
            # Copy EMA parameters to model
            self.ema_model.copy_to(self.model.parameters())
            self.best_test = True
            _state = {
                "epoch": index + 1,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            } 

            _save_path = osp.join(self.logdir, '{}_bestmodel_ema{}.pth'.format('glare',str(float(psnr_val_rgb_ema))))
            directory = os.path.dirname(_save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(_state, _save_path)

            #Restore original model parameters 
            self.ema_model.restore(self.model.parameters())

        ### Print result of PSNR###
        format_str = "epoch: {}\n avg PSNR: {:3f}, ema_avg PSNR: {:3f}"
        print_str = format_str.format(int(self.epo), float(psnr_val_rgb), float(psnr_val_rgb_ema))
        ic(print_str)
        print('--> Avg. PSNR: {:.4f}, EMA_Avg. PSNR: {:4f}'.format(float(psnr_val_rgb), float(psnr_val_rgb_ema)))
        self.logger.info(print_str)

    def train(self):
        """Start training."""
        for self.epo in range(self.start_epo, self.max_epoch):
            self.step_train('train')
            self.step_val('val')
            #if self.best_test:
            #    self.test_extend()
        self.writer.close()

    def test_extend(self):
        self.model.eval()
        
        for i, img_path in tqdm.tqdm(enumerate(self.test_feature_paths),total=len(self.test_feature_paths)):
            img = cv2.cvtColor(cv2.imread(str(img_path)),cv2.COLOR_BGR2RGB)
            in_feature = img.astype(np.float32)/255
        
            new_fname = "test_" + img_path.name[11:-4]

            H = in_feature.shape[0]
            W = in_feature.shape[1]

            finale_result = np.zeros_like(in_feature)
            extended_in_feature = np.zeros((2560,3584,3))
            extended_in_feature[:H, :W, :] = in_feature
            extended_in_feature = np.expand_dims(extended_in_feature, axis=0)
            extended_in_feature = torch.from_numpy(extended_in_feature).float().to(device=self.device)
            extended_in_feature = extended_in_feature.permute(0, 3, 1, 2)

            with torch.no_grad():
                output = self.model(extended_in_feature)

            output = torch.clamp(output,0,1)
            output = output.permute(0, 2, 3, 1).cpu().detach().numpy()

            if self.args.save_image:
                de_glared_img = output[0]*255
                finale_result = de_glared_img[:H,:W,:]
                new_fname = "test_" + img_path.name[11:-4]
                save_img(osp.join(self.args.submission_dir + new_fname + '.png'),finale_result, color_domain=self.args.color_domain)
    
    def test_crop(self):
        self.model.eval()
        
        for i, img_path in tqdm.tqdm(enumerate(self.test_feature_paths),total=len(self.test_feature_paths)):
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            in_feature = img.astype(np.float32)/255
        
            new_fname = "test_" + img_path.name[11:-4]

            H = in_feature.shape[0]
            W = in_feature.shape[1]
            STRIDE_H = 128
            STRIDE_W = 128

            extended_in_feature = np.zeros((2560,3584,3))
            extended_in_feature[:H, :W, :] = in_feature
            eH = extended_in_feature.shape[0]
            eW = extended_in_feature.shape[1]

            result_img = np.zeros_like(extended_in_feature)
            voting_mask = np.zeros_like(extended_in_feature)
            finale_result = np.zeros_like(in_feature)

            crop = []
            position = []
            batch_count = 0

            with torch.no_grad():
                for top in range(0, eH, STRIDE_H):
                    for left in range(0, eW, STRIDE_W):
                        piece = np.zeros([self.width, self.height, 3])
                        piece = extended_in_feature[top:top + self.width, left:left + self.height, :]
                        #piece[:temp.shape[0], :temp.shape[1], :] = temp
                        piece = np.expand_dims(piece, axis=0)
                        #crop.append(piece)
                        position.append([top, left])
                        batch_count += 1
                        if batch_count == 1:
                            crop = torch.from_numpy(piece).float().to(device=self.device)
                            crop = crop.permute(0, 3, 1, 2)
                            pred = self.model(crop)
                            pred = pred.permute(0, 2, 3, 1).cpu().detach().numpy()
                            pred = pred * 255.0
                            crop = []
                            batch_count = 0
                            for num, (t, l) in enumerate(position):
                                piece = pred[num]
                                #save_img(osp.join(args.submission_dir + new_fname + str(t) + '_' + str(l) + '.png'),piece)
                                #h, w, c = result_img[t:t+self.width, l:l+self.height, :].shape
                                result_img[t:t+self.width, l:l+self.height, :] += piece
                                voting_mask[t:t+self.width, l:l+self.height, :] += 1
                            position = []
                
                result_img = result_img/voting_mask
                result_img = result_img.astype(np.uint8)
                finale_result = result_img[:H,:W,:]
                
                save_img(osp.join(self.args.submission_dir + new_fname + '.png'),finale_result)
                #save_img(osp.join(args.submission_dir + new_fname + '.png'),de_glared_img)   

    def test(self):
        for index, (in_feature, fname) in tqdm.tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader)):
            in_feature = in_feature.to(self.device)
            with torch.no_grad():
                output = self.model(in_feature)
            output = torch.clamp(output,0,1)
            output = output.permute(0, 2, 3, 1).cpu().detach().numpy()
            if self.args.save_image:
                for batch in range(len(output)):
                    de_glared_img = output[batch]*255
                    new_fname = "test_" + fname[batch][11:-4]
                    cv2.imwrite(self.args.submission_dir + new_fname + '.png', cv2.cvtColor(cv2.resize(de_glared_img,(3264,2448)), cv2.COLOR_RGB2BGR))
                    #save_img(osp.join(args.submission_dir + new_fname + '.png'),cv2.resize(de_glared_img,(3264,2448)), color_domain=self.args.color_domain)


if __name__ == "__main__":
    ic.configureOutput(prefix='Deglare training |')
    ######### parser ###########
    args = options.Options().init(argparse.ArgumentParser(description='image deglare')).parse_args()
    ic(args)

    #ic.enable()
    ic.disable()

    ##### Tensorboard #####
    logdir = osp.join("./", args.work_dir)
    logger = get_logger(logdir)
    writer = SummaryWriter(log_dir=logdir)


    trainer = Trainer(args=args, logger=logger, writer=writer)
    trainer.train()
    #trainer.test_extend()
