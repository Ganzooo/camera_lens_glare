#!/usr/bin/env python3
# coding: utf-8

import argparse
import os.path as osp
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
import torchvision.models as models
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from dataset.dataloader import load_dataset
from models.nestedUnet import NestedUNet
from models.resnet50_unet_activation import UNetWithResnet50Encoder
from models.resnet50_unet_activation_drop import UNetWithResnet50Encoder_act_drop
from models.resnet50_unet_activation_no_bn import UNetWithResnet50Encoder_act_no_bn
from models.MIRNet_model import MIRNet
from models.Uformer import Uformer

from utils.utils import get_logger
from icecream import ic
from skimage import img_as_ubyte
import cv2
from utils.image_utils import save_img, batch_PSNR
import options

#import apex
#from apex.parallel import DistributedDataParallel as DDP
#from apex.fp16_utils import *
#from apex import amp, optimizers
from losses import CharbonnierLoss
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from dataset.dataset_utils import MixUp_AUG

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


class ContentLoss(nn.Module):
    def __init__(self, loss):
        super(ContentLoss, self).__init__()
        self.criterion = loss
        self.net = self.content_model()

    def get_loss(self, pred, target):
        pred_f = self.net(pred)
        target_f = self.net(target)
        loss = self.criterion(pred_f, target_f)

        # print(loss)
        return loss

    def content_model(self):
        self.cnn = models.vgg19(pretrained=True).features
        self.cnn.cuda()
        # print(self.cnn) 
        content_layers = ['relu_4']
        content_losses = []
        model = nn.Sequential()

        i = 0
        for layer in self.cnn.children():

            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                break
        
        return model


class Trainer(object):
    def __init__(self, args):
        self.data_path=args.data_path,
        self.batch_size=args.batch_size,
        self.max_epoch=args.max_epoch,
        self.pretrained_weight=args.pretrained_model,
        self.width=args.width,
        self.height=args.height,
        self.resume_train = args.resume,
        self.work_dir = args.work_dir,
        if args.random_crop:
            self.train_dataloader, self.val_dataloader, self.test_dataloader = load_dataset(args.data_path, args.batch_size, 
                    distributed=False, random_crop=True, resize_size=(args.width,args.height),
                    model_type=args.model_type,color_domain=args.color_domain)
        else:
            self.train_dataloader, self.val_dataloader, self.test_dataloader = load_dataset(args.data_path, args.batch_size, 
                    distributed=False, random_crop=False, resize_size=(args.width,args.height),
                    model_type=args.model_type,color_domain=args.color_domain)
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
        elif args.model_type == "nest":
            ic('Nested_Unet')
            self.model = NestedUNet(num_classes=3,input_channel =3, deep_supervision=self.args.deep_vision,f_activation=nn.Sigmoid()).to(self.device)
        else:
            self.model = UNetWithResnet50Encoder().to(self.device)
            ic('ResNet_Unet')
            
        print(self.model)
        ic('Total parameter of model', sum(p.numel() for p in self.model.parameters()))

        ######### Initialize Optimizer ###########
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr_initial, momentum=0.9)
         
        ######### Load Pretrained model or Resume train ###########
        if args.pretrained_model is not None:
            ic("Pretrained weight load")
            checkpoint = torch.load(args.pretrained_model)
            try:
                self.model.load_state_dict(checkpoint["model_state"])
            except:
                state_dict = checkpoint["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)

        if args.resume is not None:
            ic("Loading model and optimizer from checkpoint ")
            checkpoint = torch.load(self.resume_train)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.start_epo = checkpoint["epoch"]
            self.logger.info("Loaded checkpoint '{}' (epoch {})".format(self.resume_train, self.start_epo))
            ic(self.start_epo)
        else:
            self.logger.info("No checkpoint found at '{}'".format(self.resume_train))

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
            elif args.scheduler == 'lambda':
                self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95 ** self.max_epoch, last_epoch=-1)
            
            for i in range(1, self.start_epo):
                self.scheduler.step()
        ic("==> Training with learning rate:", self.scheduler.get_last_lr()[0])
        
        ######### Loss ###########
        if self.args.loss_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        elif self.args.loss_type == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='mean').to(self.device)
        elif self.args.loss_type == 'charb':
            self.criterion = CharbonnierLoss().to(self.device)
        else:
            self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)


        ####### Content Loss #####
        if args.content_loss:
            self.content = ContentLoss(torch.nn.MSELoss(reduction='mean'))
            self.content.net.to(self.device)
            ic('Perceptual model')
            ic(self.content.net)


    def step_train(self, mode):
        ic('Start {} -> epoch: {}'.format(mode,self.epo))
        self.model.train()

        self.best_test = False
        loss_sum = 0
        iter = 0
               
        
        for index, (img, gt, fname) in tqdm.tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            iter += 1

            img = img.to(self.device)
            gt = gt.to(self.device)

            ### Predict image ###            
            pred = self.model(img)
            #print(pred.shape) 
            if self.args.content_loss:
                if self.args.deep_vision:
                    loss = 0
                    for output in pred:
                        #print(output.shape)
                        loss += self.criterion(output, gt)
                    loss /= len(pred)
                    perceptual_loss = self.content.get_loss(pred[-1],gt)

                else:
                    loss = self.criterion(pred, gt)
                    perceptual_loss = self.content.get_loss(pred,gt)

                loss += self.args.content_lambda * perceptual_loss
                # print(loss, perceptual_loss* self.args.content_lambda, perceptual_loss)
            else:
                if self.args.deep_vision:
                    loss = 0
                    for output in pred:
                        # print(output.shape)
                        loss += self.criterion(output, gt)
                    loss /= len(pred)
                else:
                    loss = self.criterion(pred, gt)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()

            if self.args.save_image_train and np.mod(self.epo, self.args.save_print_interval) == 0:
                if iter < 20:
                    gt = gt.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
                    img = img.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
                    #print(self.args.deep_vision)    
                    if self.args.deep_vision:
                        #print(pred.shape)
                        pred = pred[-1].permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
                    else:
                        pred = pred.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
                    
                    if img.shape[0] > 4:
                        temp0 = np.concatenate((img[0]*255, pred[0]*255, gt[0]*255),axis=1)  
                        temp1 = np.concatenate((img[1]*255, pred[1]*255, gt[1]*255),axis=1)  
                        temp2 = np.concatenate((img[2]*255, pred[2]*255, gt[2]*255),axis=1)  
                        temp3 = np.concatenate((img[3]*255, pred[3]*255, gt[3]*255),axis=1)  
                        
                        #temp0 = np.concatenate((img[0]*255, (pred[0]+1)*255, gt[0]*255),axis=1)
                        #temp1 = np.concatenate((img[1]*255, (pred[1]+1)*255, gt[1]*255),axis=1)
                        #temp2 = np.concatenate((img[2]*255, (pred[2]+1)*255, gt[2]*255),axis=1)
                        #temp3 = np.concatenate((img[3]*255, (pred[3]+1)*255, gt[3]*255),axis=1)

                        temp = np.concatenate((temp0, temp1, temp2, temp3),axis=0)
                        save_img(osp.join(self.args.result_dir + 'train/' + str(self.epo) + '/batches_'+ str(iter) + '.jpg'),temp.astype(np.uint8), color_domain=self.args.color_domain)
                    else:
                        temp0 = np.concatenate((img[0]*255, pred[0]*255, gt[0]*255),axis=1)
                        #temp0 = np.concatenate((img[0]*255, (pred[0]+1)*255, gt[0]*255),axis=1) 
                        save_img(osp.join(self.args.result_dir + 'train/' + str(self.epo) + '/batches_'+ str(iter) + '.jpg'),temp0.astype(np.uint8), color_domain=self.args.color_domain)

        self.scheduler.step()
        
        ### Tensorboard ###
        avg_loss = loss_sum / iter
        avg_psnr = 10*np.log10(1/avg_loss)
        writer.add_scalar("Loss/train", avg_loss, self.epo)
        writer.add_scalar("PSNR/train", avg_psnr, self.epo)


        ### Save latest model ###
        if np.mod(self.epo, self.save_model_interval) == 0:
            _state = {
                "epoch": self.epo + 1,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            } 
            _save_path_latest = osp.join(self.logdir, '{}_latestmodel.pth'.format('glare'))
            torch.save(_state,_save_path_latest)
        
        ### Print result of loss and scheduler###
        format_str = "epoch: {}\n avg loss: {:3f}, scheduler: {}"
        print_str = format_str.format(int(self.epo) ,float(loss_sum/iter), float(self.scheduler.get_last_lr()[0]))
        ic(print_str)
        self.logger.info(print_str)

    def step_val(self, mode):
        self.model.eval()
        
        ic('Start {} -> epoch: {}'.format(mode,self.epo))
        self.best_test = False
        psnr_val_rgb = []
        iter = 0
        loss_sum = 0
        for index, (img, gt, fname) in tqdm.tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader)):
            iter += 1
            img = img.to(self.device)
            gt = gt.to(self.device)

            with torch.no_grad():
                ###Pred image ###
                pred = self.model(img)
                if self.args.deep_vision:
                    loss = 0
                    for output in pred:
                        #print(output.shape)
                        loss += self.criterion(output, gt)
                    loss /= len(pred)
                    psnr_val_rgb.append(batch_PSNR(pred[-1], gt, 1.))
                    pred = torch.clamp(pred[-1],0,1)
                    
                else:
                    loss = self.criterion(pred,gt)
                    psnr_val_rgb.append(batch_PSNR(pred, gt, 1.))
                    pred = torch.clamp(pred,0,1)
                
                loss_sum += loss.item()
                if self.args.save_image_val and np.mod(self.epo, self.args.save_print_interval) == 0:
                    gt = gt.permute(0, 2, 3, 1).cpu().contiguous().detach().numpy()
                    img = img.permute(0, 2, 3, 1).cpu().contiguous().detach().numpy()
                    pred = pred.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
                                        
                    for batch in range(img.shape[0]):
                        temp = np.concatenate((img[batch]*255.0, pred[batch]*255.0, gt[batch]*255.0),axis=1)
                        #temp = np.concatenate((img[batch]*255.0, (pred[batch]+1)*255.0, gt[batch]*255.0),axis=1)
                        save_img(osp.join(self.args.result_dir + '/val/'+ str(index) + fname[batch][:-4] +'.jpg'),temp.astype(np.uint8), color_domain=self.args.color_domain)

        psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
        
        ### Tensorboard ###
        writer.add_scalar("Loss/test", loss_sum / iter, self.epo)
        writer.add_scalar("PSNR/test", psnr_val_rgb, self.epo)

        ### Save best model ###
        if psnr_val_rgb > self.best_psnr:
            self.best_psnr = psnr_val_rgb
            self.best_test = True
            _state = {
                "epoch": index + 1,
                "model_state": self.model.state_dict(),
            } 

            _save_path = osp.join(self.logdir, '{}_bestmodel_{}.pth'.format('glare',str(float(psnr_val_rgb))))
            directory = os.path.dirname(_save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(_state, _save_path)

        ### Print result of PSNR###
        format_str = "epoch: {}\n avg PSNR: {:3f}"
        print_str = format_str.format(int(self.epo) ,float(psnr_val_rgb))
        ic(print_str)
        self.logger.info(print_str)

    def train(self):
        """Start training."""
        for self.epo in range(self.start_epo, self.max_epoch):
            self.step_train('train')
            self.step_val('val')
            #if self.best_test:
            #    self.test_extend()
            #self.test_extend()

        writer.close()
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

            if self.args.deep_vision:
                output[-1] = torch.clamp(output[-1],0,1)
                output = output[-1].permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
            else:
                output = torch.clamp(output,0,1)
                output = output.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()

            if self.args.save_image:
                de_glared_img = output[0]*255
                #de_glared_img = (output[0]+1)*255
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
                            crop = crop.permute(0, 3, 1, 2).contiguous()
                            pred = self.model(crop)
                            if self.args.deep_vision:
                                pred = pred[-1].permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
                            else:
                                pred = pred.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()

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
            if self.args.deep_vision:
                output[-1] = torch.clamp(output[-1],0,1)
                output = output[-1].permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()
            else:
                output = torch.clamp(output,0,1)
                output = output.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()

            if self.args.save_image:
                for batch in range(len(output)):
                    de_glared_img = output[batch]*255
                    #de_glared_img = (output[batch]+1)*255
                    new_fname = "test_" + fname[batch][11:-4]
                    cv2.imwrite(self.args.submission_dir + new_fname + '.png', cv2.cvtColor(cv2.resize(de_glared_img,(3264,2448)), cv2.COLOR_RGB2BGR))
                    #save_img(osp.join(args.submission_dir + new_fname + '.png'),cv2.resize(de_glared_img,(3264,2448)), color_domain=self.args.color_domain)


    def get_content_model(self, model):
        content_layer = ['conv_4']
        content_lossed = []
        pretrained_vgg = copy.deepcopy(model)
        i = 0
        for layer in pretrained_vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

                model.add_module(name, layer)

            if name in content_layers:
                target = model(pred_img).detach()
                content_loss = Content_Loss(target)
                model.add_module('Content_loss_{}'.format(i), content_loss)
                content_losses.append(content_loss)


        for i in range(len(model) -1, -1, -1):
            if isinstance(model[i], ContentLoss):
                break

        model = model[:(i+1)]
        return model, content_loss


if __name__ == "__main__":
    ic.configureOutput(prefix='Deglare training |')
    ######### parser ###########
    args = options.Options().init(argparse.ArgumentParser(description='image deglare')).parse_args()
    ic(args)

    #ic.enable()
    #ic.disable()
    
    trainer = Trainer(args=args)
    trainer.train()
    #trainer.test_extend()
