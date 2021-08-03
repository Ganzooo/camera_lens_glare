
from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse
import math 

from skimage import measure

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--gt_dir', default='/dataset_sub/camera_light_glare/TEST_IMAGE_SIMILAR/test_gt/', type=str, help='Directory for full resolution images')
parser.add_argument('--pred_dir', default='/dataset_sub/camera_light_glare/TEST_IMAGE_SIMILAR/test_pred/tensorrt/',type=str, help='Directory for image patches')

args = parser.parse_args()

#get sorted folders
files_gt = natsorted(glob(os.path.join(args.gt_dir, '*.png')))
files_pred = natsorted(glob(os.path.join(args.pred_dir, '*.png')))
gt_files, pred_files = [], []
for file_ in files_gt:
    #filename = os.path.split(file_)[-1]
    gt_files.append(file_)

for file_ in files_pred:
    #filename = os.path.split(file_)[-1]
    pred_files.append(file_)

psnr_rgb = []
psnr_rgb2 = []

def rmse_score(true, pred):
    score = math.sqrt(np.mean((true-pred)**2))
    return score

def psnr_score(true, pred, pixel_max=255):
    score = 20*np.log10(pixel_max/rmse_score(true, pred))
    return score

for i in tqdm(range(len(gt_files))):
    gt_file, pred_file = gt_files[i], pred_files[i]
    gt_img = cv2.imread(gt_file)
    pred_img = cv2.imread(pred_file)
    psnr_rgb.append(measure.compare_psnr(gt_img, pred_img, data_range=255))
    psnr_rgb2.append(psnr_score(gt_img.astype(float), pred_img.astype(float)))

print(psnr_rgb)
print(psnr_rgb2)
print("average psnr: ", sum(psnr_rgb) / len(psnr_rgb))
print("average psnr2: ", sum(psnr_rgb2) / len(psnr_rgb))