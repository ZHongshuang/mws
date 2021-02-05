# coding=utf-8
import pdb
import torch
from tqdm import tqdm
from networks.sal import SalModel
from datasets.saliency import Folder
from evaluate_sal import fm_and_mae
import json
import os
import pdb
import random


def test(model, val_loader, opt):
    print("============================= TEST ============================")
    model.switch_to_eval()
    for i, (img, name, WW, HH) in tqdm(enumerate(val_loader), desc='testing'):
        model.test(img, name, WW, HH)
    model.switch_to_train()
    maxfm, mae, _, _ = fm_and_mae(opt.results_dir, opt.val_gt_dir)
    model.performance = {'maxfm': maxfm, 'mae': mae}
    return model.performance


class CombinedIter(object):
    def __init__(self, sal_loader, syn_loader):
        self.sal_loader = sal_loader
        self.sal_iter = iter(sal_loader)
        self.sal_i = 0

        self.syn_loader = syn_loader
        self.syn_iter = iter(syn_loader)
        self.syn_i = 0

    def next(self):
        if self.sal_i >= len(self.sal_loader):
            self.sal_iter = iter(self.sal_loader)
            self.sal_i = 0
        img_sal, gt_sal = self.sal_iter.next()
        self.sal_i += 1

        if self.syn_i >= len(self.syn_loader):
            self.syn_iter = iter(self.syn_loader)
            self.syn_i = 0
        img_syn, gt_syn = self.syn_iter.next()
        self.syn_i += 1

        output = {'img_sal': img_sal, 'gt_sal': gt_sal.unsqueeze(1),
                  'img_syn': img_syn, 'gt_syn': gt_syn.unsqueeze(1)}
        return output


def train(model, train_loader, syn_loader, val_loader, opt):
    print("============================= TRAIN ============================")
    model.switch_to_train()
    # model.load('best')

    train_iter = CombinedIter(train_loader, syn_loader)
    log = {'best': 0, 'best_it': 0}

    for i in tqdm(range(opt.train_iters), desc='train'):
        data = train_iter.next()
        pdb.set_trace()

        model.set_input(data)
        model.optimize_parameters(i)

        # if i % opt.display_freq == 0:
        #     model.show_tensorboard(i)

        if i != 0 and i % opt.save_latest_freq == 0:
            model.save(i)
            performance = test(model, val_loader, opt)
            model.show_tensorboard_eval(i)
            log[i] = performance
            if performance['maxfm'] > log['best']:
                log['best'] = performance['maxfm']
                log['best_it'] = i
                model.save('best')
            print(u'最大fm: iter%d的%.4f' % (log['best_it'], log['best']))
            for k, v in performance.items():
                print(u'这次%s: %.4f' % (k, v))
            with open(model.save_dir + '/' + 'train-log.json', 'w') as outfile:
                json.dump(log, outfile)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=256, help='input image size')
    parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406], help='input image size')
    parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225], help='input image size')
    parser.add_argument('--name', type=str,
                             help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--model', type=str, help='which model to use')
    parser.add_argument('--results_dir', type=str, default='./results',
                             help='path to save validation results.')
    parser.add_argument('--base', type=str, default='densenet169',
                             help='chooses which backbone network to use. densenet169, vgg16, etc')

    parser.add_argument('--checkpoints_dir', type=str, default='./mwsFiles',
                             help='path to save params and tensorboard files')

    parser.add_argument('--start_it', type=int, default=0, help='recover from saved')
    parser.add_argument('--display_freq', type=int, default=20,
                             help='frequency of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=100,
                             help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of saving the latest model')
    parser.add_argument('--train_iters', type=int, default=5000, help='training iterations')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--is_train', type=bool, default=True, help='train, val, test, etc')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')

    parser.add_argument('--train_img_dir', type=str,
                             default='/home/data3/zhang.hongshuang/Data/saliency/DUTS-TR/images',
                             help='path to saliency training images ')
    parser.add_argument('--train_gt_dir', type=str,
                             default='/home/data3/zhang.hongshuang/Data/saliency/DUTS-TR/masks')# /DUT-train_two_mr2_crf_bin
    parser.add_argument('--val_img_dir', type=str,
                             default='/home/data3/zhang.hongshuang/Data/saliency/ECSSD/images',
                             help='path to validation images')
    parser.add_argument('--val_gt_dir', type=str,
                             default='/home/data3/zhang.hongshuang/Data/saliency//ECSSD/masks',
                             help='path to validation ground-truth')
    parser.add_argument('--syn_img_dir', type=str,
                             default='/home/data3/zhang.hongshuang/Data/saliency/DUTS-TR/images',
                             help='path to validation images')
    parser.add_argument('--syn_gt_dir', type=str,
                             default='/home/data3/zhang.hongshuang/Data/saliency/DUTS-TR/masks',
                             help='path to validation ground-truth')

    opt = parser.parse_args()
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)

    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)



    val_loader = torch.utils.data.DataLoader(
        Folder(opt.val_img_dir, opt.val_gt_dir,
               crop=None, flip=False, rotate=None, size=opt.imageSize,
               mean=opt.mean, std=opt.std, training=False),
        batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
        Folder(opt.train_img_dir, opt.train_gt_dir,
               crop=0.9, flip=True, rotate=None, size=opt.imageSize,
               mean=opt.mean, std=opt.std, training=True),
        batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)

    syn_loader = torch.utils.data.DataLoader(
        Folder(opt.syn_img_dir, opt.syn_gt_dir,
               crop=0.9, flip=True, rotate=None, size=opt.imageSize,
               mean=opt.mean, std=opt.std, training=True),
        batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)

    model = SalModel(opt)

    train(model, train_loader, syn_loader, val_loader, opt)
    print("We are done")



