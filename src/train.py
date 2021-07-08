#!/usr/bin/env python3
# coding: utf-8

import argparse
import os.path as osp
import sys

import gdown
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import visdom
from datetime import datetime
import random

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from dataset.dataloader import load_dataset
from optimizers import get_optimizer
from schedulers import get_scheduler
from models.resnet50_unet import UNetWithResnet50Encoder
#from models.MIRNet_model import MIRNet

from utils.utils import get_logger
from icecream import ic
from skimage import img_as_ubyte
import cv2
from utils.image_utils import save_img, batch_PSNR

import apex
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from losses import CharbonnierLoss
from warmup_scheduler import GradualWarmupScheduler
from dataset.dataset_utils import MixUp_AUG

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)


class Trainer(object):
    def __init__(self, data_path, batch_size, max_epoch, pretrained_model,
                 train_data_num, val_data_num,
                 width, height, use_constant_feature, use_intensity_feature, vis_on, resume_train, work_dir, args):

        self.train_dataloader, self.val_dataloader, self.test_dataloader = load_dataset(data_path, batch_size, distributed=False, train_valid_split_weight=0.9, resize_size=(args.width,args.height))
        self.max_epoch = max_epoch
        self.time_now = datetime.now().strftime('%Y%m%d_%H%M')
        self.best_psnr = 1e10
        self.args = args

        self.logdir = osp.join("./", work_dir)
        self.logger = get_logger(self.logdir)
        self.logger.info("Let the train begin...")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNetWithResnet50Encoder().to(self.device)

        self.save_model_interval = 1
        self.loss_print_interval = 1
        self.mixup = MixUp_AUG()

        #self.optimizer = get_optimizer("SGD", self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)
        ######### Scheduler ###########
        warmup = True
        if warmup:
            warmup_epochs = 3
            scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch-warmup_epochs, eta_min=1e-6)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
            self.scheduler.step()

        #self.scheduler = get_scheduler('LambdaLR', self.optimizer, self.max_epoch)

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                        opt_level=args.opt_level)

        total_params = sum(p.numel() for p in self.model.parameters())
        #print( 'Parameters:',total_params )
        ic('Total Parameters:',total_params)

        self.start_epo = 0
        self.resume_train = resume_train
        if resume_train is not None:
            self.logger.info("Loading model and optimizer from checkpoint ")
            #print("Loading model and optimizer from checkpoint ")
            ic("Loading model and optimizer from checkpoint ")
            checkpoint = torch.load(resume_train)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.start_epo = checkpoint["scheduler_state"]["last_epoch"]
            self.logger.info("Loaded checkpoint '{}' (epoch {})".format(resume_train, self.start_epo))
        else:
            self.logger.info("No checkpoint found at '{}'".format(resume_train))

    def step(self, mode):
        """Proceed with training or verification

        Parameters
        ----------
        mode : str
            Specify training or verification. 'train' or 'val'

        """
        #print('Start {}'.format(mode))
        ic('Start {}'.format(mode))

        if mode == 'train':
            self.model.train()
            dataloader = self.train_dataloader
        elif mode == 'val':
            self.model.eval()
            dataloader = self.val_dataloader

        loss_sum = 0

        #criterion = torch.nn.MSELoss().to(self.device)
        ######### Loss ###########
        criterion = CharbonnierLoss().cuda()
        for index, (img, gt, fname) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            
            img = img.to(self.device)
            gt = gt.to(self.device)

            if mode == 'train':
                if self.epo > 5:
                    target, input_ = self.mixup.aug(gt, img)
                pred = self.model(img)
            
                loss = criterion(pred, gt)

                self.optimizer.zero_grad()
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                #loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()
            
            if mode == 'val':
                psnr_val_rgb = []
                with torch.no_grad():
                    pred = self.model(img)
                    pred = torch.clamp(pred,0,1) 
                    psnr_val_rgb.append(batch_PSNR(pred, gt, 1.))

                    if args.save_image:
                        gt = gt.permute(0, 2, 3, 1).cpu().detach().numpy()
                        img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
                        pred = pred.permute(0, 2, 3, 1).cpu().detach().numpy()
                        
                        for batch in range(img.shape[0]):
                            temp = np.concatenate((img[batch]*255, pred[batch]*255, gt[batch]*255),axis=1)
                            save_img(osp.join(args.result_dir + str(index) + fname[batch][:-4] +'.jpg'),temp.astype(np.uint8))

        if mode == 'val':
            psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
            if psnr_val_rgb > self.best_psnr:
                ic('update best model {} -> {}'.format(self.best_loss, loss_sum / len(dataloader)))
                _state = {
                    "epoch": index + 1,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                } 
                _save_path = osp.join(self.logdir, '{}_bestmodel.pt'.format('glare'))
                torch.save(_state, _save_path)
            format_str = "epoch: {}\n avg PSNR: {:3f}"
            print_str = format_str.format(int(self.epo) ,float(psnr_val_rgb))
            ic(print_str)
            self.logger.info(print_str)

        if np.mod(index, self.loss_print_interval) == 0:
            format_str = "epoch: {}\n avg loss: {:3f}"
            print_str = format_str.format(int(self.epo) ,float(loss_sum))
            ic(print_str)
            self.logger.info(print_str)

        if np.mod(self.epo, self.save_model_interval) == 0:
            _state = {
                "epoch": index + 1,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
            } 
            _save_path_latest = osp.join(self.logdir, '{}_latestmodel_{}.pt'.format('glare',index+1))
            torch.save(_state,_save_path_latest)

    def train(self):
        """Start training."""
        for self.epo in range(self.start_epo, self.max_epoch):
            self.step('train')
            self.step('val')
            self.scheduler.step()

    def test(self):
        """start testing."""
        if self.resume_train == None:
            ic("need to reload trained weight")
            return 0

        for index, (in_feature, fname) in tqdm.tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader)):
            
            with torch.no_grad():
                output = self.model(in_feature)
            output = torch.clamp(output,0,1)
            output = output.permute(0, 2, 3, 1).cpu().detach().numpy()

            if args.save_image:
                for batch in range(len(output)):
                    de_glared_img = img_as_ubyte(output[batch])
                    cv2.imwrite(args.submission_dir + fname[batch][:-4] + '.png', de_glared_img)

        

def parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--data_path',
        '-dp',
        type=str,
        help='Training data path',
        #default='/dataset/nuScenes/FeatureExtracted/v1.0-trainval/')
        default='/dataset_sub/camera_light_glare/')
    parser.add_argument('--batch_size', '-bs', type=int,
                        help='batch size',
                        default=6)
    parser.add_argument('--max_epoch', '-me', type=int,
                        help='max epoch',       
                        default=300)
    parser.add_argument('--pretrained_model', '-p', type=str,
                        help='Pretrained model path',
                        default='./checkpoints/bcnn_latestmodel_20210607_0809.pt')
    parser.add_argument('--train_data_num', '-tn', type=int,
                        help='Number of data used for training. Larger number if all data are used.',
                        default=1000000)
    parser.add_argument('--val_data_num', '-vn', type=int,
                        help='Nuber of  data used for validation. Larger number if all data are used.',
                        default=100000)
    parser.add_argument('--width', type=int,
                        help='feature map width',
                        default=512)
    parser.add_argument('--height', type=int,
                        help='feature map height',
                        default=512)
    parser.add_argument('--use_constant_feature', type=int,
                        help='Whether to use constant feature',
                        default=0)
    parser.add_argument('--use_intensity_feature', type=int,
                        help='Whether to use intensity feature',
                        default=0)
    parser.add_argument('--visualization_on', type=int,
                        help='Whether to use visualaziation on during train',
                        default=0)
    parser.add_argument('--resume', type=str,
                        help='Train process resume cur/bcnn_latestmodel.pt',
                        #default='./cur/deglare/bcnn_latestmodel.pt')
                        default=None)
    parser.add_argument('--work_dir', type=str,
                        help='Work directory cur/bcnn',
                        default='./cur/deglare')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--sync-bn', action='store_true',
                        help='enabling apex sync BN.', default =False)

    parser.add_argument('--opt-level', type=str, default = 'O1')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--loss_type', type=str, default='BcnnLoss')
    parser.add_argument('--save_image', type=bool, default=True)
    parser.add_argument('--submission_dir', type=str,
                        help='Work directory submission',
                        default='./submission/')
    parser.add_argument('--result_dir', type=str,
                        help='Work directory submission',
                        default='./result/')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    
    ic.configureOutput(prefix='CNN training |')
    ic.enable()
    #ic.disable()

    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    
    trainer = Trainer(data_path=args.data_path,
                      batch_size=args.batch_size,
                      max_epoch=args.max_epoch,
                      pretrained_model=args.pretrained_model,
                      train_data_num=args.train_data_num,
                      val_data_num=args.val_data_num,
                      width=args.width,
                      height=args.height,
                      use_constant_feature=args.use_constant_feature,
                      use_intensity_feature=args.use_intensity_feature,
                      vis_on = args.visualization_on,
                      resume_train = args.resume,
                      work_dir = args.work_dir, 
                      args=args)
    trainer.train()
    trainer.test()
