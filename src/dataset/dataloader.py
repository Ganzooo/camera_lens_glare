#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path
from albumentations.augmentations.functional import gamma_transform
from albumentations.augmentations.transforms import ColorJitter

import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch
import cv2
import torchvision
from augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale, RandomVerticallyFlip
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_dataset(data_path, batch_size, distributed, center_crop=False, random_crop=False, h2g_aug=False, resize_size=(512,512), model_type='baseline', color_domain='rgb', tv_change=False):
    if random_crop:
        transformer_train = A.Compose([
            #A.Resize(resize_size[0],resize_size[1]),
            A.RandomCrop(width=resize_size[0], height=resize_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()
            ], additional_targets={'target': 'image'})
    elif center_crop:
        transformer_train = A.Compose([
            #A.Resize(resize_size[0],resize_size[1]),
            #A.RandomCrop(width=resize_size[0], height=resize_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()
            ], additional_targets={'target': 'image'})
    else: 
        transformer_train = A.Compose([
            #A.Resize(resize_size[0],resize_size[1]),
            #A.RandomCrop(width=resize_size[0], height=resize_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()
            ], additional_targets={'target': 'image'})

    if h2g_aug:
        transformer_train = A.Compose([
            A.RandomCrop(width=resize_size[0], height=resize_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(p=0.5, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ToTensorV2()
            ], additional_targets={'target': 'image'})

    transformer_valid = A.Compose([
            A.Resize(resize_size[0],resize_size[1]),
            ToTensorV2()
            ], additional_targets={'target': 'image'})

    if model_type == "MIRNet":
        transformer_test = A.Compose([A.Resize(2048, 1024), ToTensorV2()])
    elif model_type == "Uformer16":
        transformer_test = A.Compose([A.Resize(2048, 2048), ToTensorV2()])
    elif model_type == "Uformer32":
        transformer_test = A.Compose([A.Resize(2048, 2048), ToTensorV2()])
    else:
        transformer_test = A.Compose([A.Resize(3584, 2560), ToTensorV2()])

    train_dataset = DataLoaderImg(data_path, mode='train', resize_size=resize_size, transform=transformer_train, color_domain=color_domain, tv_change=tv_change)
    val_dataset = DataLoaderImg(data_path, mode='val', resize_size=resize_size, transform=transformer_valid, color_domain=color_domain, tv_change=tv_change)
    test_dataset = DataLoaderImg(data_path, mode='test', transform=transformer_test, color_domain=color_domain)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=True, num_workers=16, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


class DataLoaderImg(Dataset):
    def __init__(self, data_path, mode='train', transform=True, resize_size=(512,512), color_domain='rgb', tv_change=False):
        self.mode = mode
        self.data_path = data_path
        self.transform = transform
        self.target_size = resize_size
        self.dataset_type = mode
        self.color_domain = color_domain
        augmentations = Compose([RandomHorizontallyFlip(0.5), RandomVerticallyFlip(0.5)])
        self.augmentations = augmentations

        if self.dataset_type == 'train':
            #self.in_feature_paths = list(sorted(Path(self.data_path).glob("train/train_input_img/*.png")))
            #self.target_feature_paths = list(sorted(Path(self.data_path).glob("train/train_label_img/*.png")))
            if tv_change:
                self.in_feature_paths = list(sorted(Path(self.data_path).glob("val_patch/train_input_img/*.png")))
                self.target_feature_paths = list(sorted(Path(self.data_path).glob("val_patch/train_label_img/*.png")))
            else:
                self.in_feature_paths = list(sorted(Path(self.data_path).glob("train_patch/train_input_img/*.png")))
                self.target_feature_paths = list(sorted(Path(self.data_path).glob("train_patch/train_label_img/*.png")))
        elif self.dataset_type == 'val':
            #self.in_feature_paths = list(sorted(Path(self.data_path).glob("train/train_input_img/*.png")))
            #self.target_feature_paths = list(sorted(Path(self.data_path).glob("train/train_label_img/*.png")))
            if tv_change:
                self.in_feature_paths = list(sorted(Path(self.data_path).glob("train_patch/train_input_img/*.png")))
                self.target_feature_paths = list(sorted(Path(self.data_path).glob("train_patch/train_label_img/*.png")))
            else:
                self.in_feature_paths = list(sorted(Path(self.data_path).glob("val_patch/train_input_img/*.png")))
                self.target_feature_paths = list(sorted(Path(self.data_path).glob("val_patch/train_label_img/*.png"))) 
        elif self.dataset_type == 'test':
            self.test_feature_paths = list(sorted(Path(self.data_path).glob("test_input_img/*.png")))

    def __len__(self):
        if self.dataset_type == 'test':
            return len(self.test_feature_paths)
        return len(self.in_feature_paths)
    
    def __transform__(self, img):
        if self.mode == 'test':
            img = cv2.resize(img, (2048, 1024))
        
        img = cv2.resize(img, (self.target_size[0], self.target_size[1]))
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).float()

    def __getitem__(self, idx):
        if self.dataset_type == 'test':
            img = cv2.cvtColor(cv2.imread(str(self.test_feature_paths[idx])), cv2.COLOR_BGR2RGB)
            #data = self.transform(image=img)
            #img = (data['image'] / 255.0)
            img = img / 255.0
            return img, self.test_feature_paths[idx].name

        if self.color_domain == 'ycbcr':
            img = cv2.cvtColor(cv2.imread(str(self.in_feature_paths[idx])), cv2.COLOR_BGR2YCR_CB)
            lbl = cv2.cvtColor(cv2.imread(str(self.target_feature_paths[idx])), cv2.COLOR_BGR2YCR_CB)
        else: 
            img = cv2.cvtColor(cv2.imread(str(self.in_feature_paths[idx])), cv2.COLOR_BGR2RGB)
            lbl = cv2.cvtColor(cv2.imread(str(self.target_feature_paths[idx])), cv2.COLOR_BGR2RGB)

        # if self.augmentations is not None:
        #     img, lbl = self.augmentations(img, lbl)
        # return self.__transform__(img), self.__transform__(lbl), self.in_feature_paths[idx].name

        # albumentations
        data = self.transform(image=img, target=lbl)
        img = (data['image'] / 255.0)
        lbl = (data['target'] / 255.0)
        return img, lbl, self.in_feature_paths[idx].name
