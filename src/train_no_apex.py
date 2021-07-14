#!/usr/bin/env python3
# coding: utf-8

import argparse
import os.path as osp
import sys
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
from datetime import datetime
import random
from collections import OrderedDict

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from dataset.dataloader import load_dataset
from models.resnet50_unet import UNetWithResnet50Encoder
from models.MIRNet_model import MIRNet
from models.Uformer import Uformer

from utils.utils import get_logger
from icecream import ic
from skimage import img_as_ubyte
import cv2
from utils.image_utils import save_img, batch_PSNR

#import apex
#from apex.parallel import DistributedDataParallel as DDP
#from apex.fp16_utils import *
#from apex import amp, optimizers
from losses import CharbonnierLoss
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from dataset.dataset_utils import MixUp_AUG

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)

class Trainer(object):
    def __init__(self, data_path, batch_size, max_epoch, pretrained_model,
                 width, height, resume_train, work_dir, args):

        self.train_dataloader, self.val_dataloader, self.test_dataloader = load_dataset(data_path, batch_size, distributed=False, train_valid_split_weight=0.9, resize_size=(args.width,args.height),model_type=args.model_type,color_domain=args.color_domain)
        self.max_epoch = max_epoch
        self.time_now = datetime.now().strftime('%Y%m%d_%H%M')
        self.best_psnr = 0
        self.best_test = False
        self.args = args

        self.logdir = osp.join("./", work_dir)
        self.logger = get_logger(self.logdir)
        self.logger.info("Let the train begin...")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.model_type == "MIRNet":
            self.model = MIRNet().to(self.device)
            ic('MIRNet')
        elif args.model_type == "Uformer16":
            self.model = Uformer(img_size=width,embed_dim=16,win_size=8,token_embed='linear',token_mlp='leff').to(self.device)
            ic('Uformer16')
        elif args.model_type == "Uformer32":
            ic('Uformer32')
            self.model = Uformer(img_size=width,embed_dim=32,win_size=8,token_embed='linear',token_mlp='leff').to(self.device)
        else:
            self.model = UNetWithResnet50Encoder().to(self.device)
            ic('ResNet_Unet')

        self.save_model_interval = 1
        self.loss_print_interval = 1
        self.mixup = MixUp_AUG()

        #self.optimizer = get_optimizer("SGD", self.model)
        new_lr = 2e-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        #print( 'Parameters:',total_params )
        ic('Total Parameters:',total_params)
        ic('Loss type:',args.loss_type)
        
        self.pretrained_weight= pretrained_model
        if pretrained_model is not None:
            ic("Pretrained weight load")
            checkpoint = torch.load(pretrained_model)
            try:
                self.model.load_state_dict(checkpoint["state_dict"])
            except:
                state_dict = checkpoint["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)

        self.start_epo = 0
        self.resume_train = resume_train
        if resume_train is not None:
            ic("Loading model and optimizer from checkpoint ")
            checkpoint = torch.load(resume_train)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            #self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.start_epo = checkpoint["scheduler_state"]["last_epoch"]
            self.logger.info("Loaded checkpoint '{}' (epoch {})".format(resume_train, self.start_epo))
            ic(self.start_epo)
            # for i in range(1, self.start_epo):
            # #     self.scheduler.step()
            # new_lr = self.scheduler.get_last_lr()[0]
            # ic("==> Resuming Training with learning rate:", new_lr)
        else:
            self.logger.info("No checkpoint found at '{}'".format(resume_train))

        ######### Scheduler ###########
        warmup = False
        if warmup:
           warmup_epochs = self.start_epo
           scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch-warmup_epochs, eta_min=1e-6)
           self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
           self.scheduler.step()
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch, eta_min=1e-6)
            for i in range(1, self.start_epo):
                self.scheduler.step()
           #self.scheduler.step()
        new_lr = self.scheduler.get_last_lr()[0]
        ic("==> Training with learning rate:", new_lr)

    def step(self, mode):
        #print('Start {}'.format(mode))
        ic('Start {} -> epoch: {}'.format(mode,self.epo))
        self.best_test = False

        if mode == 'train':
            self.model.train()
            dataloader = self.train_dataloader
        elif mode == 'val':
            self.model.eval()
            dataloader = self.val_dataloader

        loss_sum = 0
        psnr_val_rgb = []

        ######### Loss ###########
        if self.args.loss_type == 'mse':
            criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        elif self.args.loss_type == 'l1':
            criterion = torch.nn.L1Loss(reduction='mean').to(self.device)
        elif self.args.loss_type == 'charb':
            criterion = CharbonnierLoss().to(self.device)
        else:
            criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        
        iter = 0
        for index, (img, gt, fname) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            
            img = img.to(self.device)
            gt = gt.to(self.device)

            if mode == 'train':
                iter += 1
                pred = self.model(img)
            
                loss = criterion(pred, gt)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()

                if args.save_image:
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
                            save_img(osp.join(args.result_dir + 'train/' + str(self.epo) + '/batches_'+ str(iter) + '.jpg'),temp.astype(np.uint8), color_domain=self.args.color_domain)
                        else:
                            temp0 = np.concatenate((img[0]*255, pred[0]*255, gt[0]*255),axis=1)
                            save_img(osp.join(args.result_dir + 'train/' + str(self.epo) + '/batches_'+ str(iter) + '.jpg'),temp0.astype(np.uint8), color_domain=self.args.color_domain)

            if mode == 'val':
                with torch.no_grad():
                    pred = self.model(img)
                    pred = torch.clamp(pred,0,1) 
                    psnr_val_rgb.append(batch_PSNR(pred, gt, 1.))

                    if args.save_image:
                        gt = gt.permute(0, 2, 3, 1).cpu().detach().numpy()
                        img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
                        pred = pred.permute(0, 2, 3, 1).cpu().detach().numpy()
                        
                        for batch in range(img.shape[0]):
                            temp = np.concatenate((img[batch]*255.0, pred[batch]*255.0, gt[batch]*255.0),axis=1)
                            save_img(osp.join(args.result_dir + '/val/'+ str(index) + fname[batch][:-4] +'.jpg'),temp.astype(np.uint8))

        self.scheduler.step()
        if np.mod(self.epo, self.loss_print_interval) == 0 and mode=='train':
            format_str = "epoch: {}\n avg loss: {:3f}, scheduler: {}"
            print_str = format_str.format(int(self.epo) ,float(loss_sum/iter), float(self.scheduler.get_last_lr()[0]))
            ic(print_str)
            self.logger.info(print_str)
        if mode == 'val':
            psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
            if psnr_val_rgb > self.best_psnr:
                self.best_psnr = psnr_val_rgb
                self.best_test = True
                #ic('update best model {} -> {}'.format(self.best_loss, loss_sum / len(dataloader)))
                _state = {
                    "epoch": index + 1,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                } 

                _save_path = osp.join(self.logdir, '{}_bestmodel_{}.pth'.format('glare',str(float(psnr_val_rgb))))
                directory = os.path.dirname(_save_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save(_state, _save_path)
            format_str = "epoch: {}\n avg PSNR: {:3f}"
            print_str = format_str.format(int(self.epo) ,float(psnr_val_rgb))
            ic(print_str)
            self.logger.info(print_str)

        if np.mod(self.epo, self.save_model_interval) == 0:
            _state = {
                "epoch": self.epo + 1,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
            } 
            _save_path_latest = osp.join(self.logdir, '{}_latestmodel.pth'.format('glare'))
            torch.save(_state,_save_path_latest)
        #self.scheduler.step()

    def train(self):
        """Start training."""
        for self.epo in range(self.start_epo, self.max_epoch):
            self.step('train')
            self.step('val')
            if self.best_test:
                self.test()
            
    def test(self):
        for index, (in_feature, fname) in tqdm.tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader)):
            in_feature = in_feature.to(self.device)
            with torch.no_grad():
                output = self.model(in_feature)
            output = torch.clamp(output,0,1)
            output = output.permute(0, 2, 3, 1).cpu().detach().numpy()
            if args.save_image:
                for batch in range(len(output)):
                    de_glared_img = output[batch]*255
                    new_fname = "test_" + fname[batch][11:-4]
                    cv2.imwrite(args.submission_dir + new_fname + '.png', cv2.cvtColor(cv2.resize(de_glared_img,(3264,2448)), cv2.COLOR_RGB2BGR), color_domain=self.args.color_domain)
                    #cv2.imwrite(args.submission_dir + fname[batch][:-4] + '.png', cv2.cvtColor(cv2.resize(img,(3264,2448)), cv2.COLOR_RGB2BGR))

def parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--data_path',
        '-dp',
        type=str,
        help='Training data path',
        default='/dataset_sub/camera_light_glare/patches_512_std2/')
    parser.add_argument('--batch_size', '-bs', type=int,
                        help='batch size',
                        default=2)
    parser.add_argument('--max_epoch', '-me', type=int,
                        help='max epoch',       
                        default=300)
    parser.add_argument('--pretrained_model', '-p', type=str,
                        help='Pretrained model path',
                        #default='./checkpoints/model_denoising.pth')
                        default=None)
    parser.add_argument('--width', type=int,
                        help='feature map width',
                        default=512)
    parser.add_argument('--height', type=int,
                        help='feature map height',
                        default=512)
    parser.add_argument('--resume', type=str,
                        help='Train process resume cur/bcnn_latestmodel.pt',
                        #default='/workspace/NETWORK/camera_lens_glare/checkpoints/glare_bestmodel_25.205883026123047.pth')
                        #default= './cur/deglare_resnet_300/glare_latestmodel.pth')
                        default=None)
    parser.add_argument('--work_dir', type=str,
                        help='Work directory cur/bcnn',
                        default='./cur/deglare_rgb')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--sync-bn', action='store_true',
                        help='enabling apex sync BN.', default =False)
    parser.add_argument('--opt-level', type=str, default = 'O1')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--loss_type', type=str, help='mse, l1, charb', default='mse')
    parser.add_argument('--save_image', type=bool, default=True)
    parser.add_argument('--submission_dir', type=str, help='Work directory submission', default='./submission/')
    parser.add_argument('--result_dir', type=str, help='Work directory submission', default='./result/')
    parser.add_argument('--model_type', type=str, help='Work directory submission', default='resnet_unet')
    parser.add_argument('--color_domain', type=str, help='color_domian, rgb, ycbcr', default='ycbcr')
    

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    
    ic.configureOutput(prefix='CNN training |')
    ic.enable()
    #ic.disable()
    print(args.data_path)
    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    
    trainer = Trainer(data_path=args.data_path,
                      batch_size=args.batch_size,
                      max_epoch=args.max_epoch,
                      pretrained_model=args.pretrained_model,
                      width=args.width,
                      height=args.height,
                      resume_train = args.resume,
                      work_dir = args.work_dir, 
                      args=args)
    trainer.train()
    #trainer.test()
