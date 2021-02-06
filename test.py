# coding=utf-8

import pdb
import time
import torch
import sys
from tqdm import tqdm
from datasets.saliency import Folder
from evaluate_sal import fm_and_mae
import pickle
import json
import os
import random
from torchvision import transforms

from networks.sal import SalModel



def test(model):
    print("============================= TEST ============================")
    model.switch_to_eval()
    for i, (img, name, WW, HH) in tqdm(enumerate(val_loader), desc='testing'):
        model.test(img, name, WW, HH)
    model.switch_to_train()
    maxfm, mae, _, _ = fm_and_mae(opt.results_dir, opt.val_gt_dir)
    model.performance = {'maxfm': maxfm, 'mae': mae}
    return model.performance





if __name__ == "__main__":
    test_dataset = 'ECSSD'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--is_train', type=bool, default=False, help='train, val, test, etc')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
    parser.add_argument('--val_img_dir', type=str,
                             default='/home/Data/saliency/%s/images' %test_dataset,
                             help='path to validation images')
    parser.add_argument('--val_gt_dir', type=str,
                             default='/home/Data/saliency/%s/masks'%test_dataset,
                             help='path to validation ground-truth')

    parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=256, help='input image size')
    parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406], help='input image size')
    parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225], help='input image size')
    parser.add_argument('--name', type=str,
                             help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--model', type=str, help='which model to use')
    parser.add_argument('--results_dir', type=str, default='./results/%s'%test_dataset,
                             help='path to save validation results.')
    parser.add_argument('--base', type=str, default='densenet169',
                             help='chooses which backbone network to use. densenet169, vgg16, etc')
    parser.add_argument('--checkpoints_dir', type=str, default='./mwsFiles',
                             help='path to save params and tensorboard files')

    opt = parser.parse_args()

    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)

    val_loader = torch.utils.data.DataLoader(
        Folder(opt.val_img_dir, opt.val_gt_dir,
               crop=None, flip=False, rotate=None, size=opt.imageSize,
               mean=opt.mean, std=opt.std, training=False),
        batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)

    model = SalModel(opt)

    model.switch_to_eval()
    model.load('best')
    performance = test(model)
    for k, v in performance.items():
        print(u'这次%s: %.4f' % (k, v))

    print("We are done")
