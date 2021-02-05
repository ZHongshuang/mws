# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
import pdb
import os

from .densenet import *




def proc_densenet(model):
    # dilation
    def remove_sequential(all_layers, network):
        for layer in network.children():
            if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(all_layers, layer)
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)

    model.features.transition2[-1].kernel_size = 1
    model.features.transition2[-1].stride = 1
    all_layers = []
    remove_sequential(all_layers, model.features.denseblock3)
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            m.dilation = (2, 2)
            m.padding = (2, 2)

    model.features.transition3[-1].kernel_size = 1
    model.features.transition3[-1].stride = 1
    all_layers = []
    remove_sequential(all_layers, model.features.denseblock4)
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            m.dilation = (4, 4)
            m.padding = (4, 4)
    model.classifier = None
    return model

procs = {
    'densenet169': proc_densenet,
}


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.ConvTranspose2d) and m.in_channels == m.out_channels:
        initial_weight = get_upsampling_weight(
            m.in_channels, m.out_channels, m.kernel_size[0])
        m.weight.data.copy_(initial_weight)


def fraze_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.weight.requires_grad=False
        m.requires_grad=False

dim_dict = {
    'densenet169': [64, 128, 256, 640, 1664],
    'vgg16': [64, 128, 256, 512, 512],
    'resnet101': [64, 256, 512, 1024, 2048]
}

class DeepLab(nn.Module):
    def __init__(self, pretrained=True, c_output=21, base='densenet169'):
        super(DeepLab, self).__init__()
        dims = dim_dict[base][-1]
        self.multi_preds = nn.ModuleList([
            nn.Conv2d(dims, c_output, kernel_size=3, dilation=dl, padding=dl) for dl in [6, 12, 18, 24]
        ])
        self.upscale = nn.ConvTranspose2d(c_output, c_output, 16, 8, 4)
        self.apply(weight_init)
        self.feature = densenet169(pretrained=pretrained)
        self.feature = procs[base](self.feature)
        self.apply(fraze_bn)

    def forward(self, x):
        x = self.feature(x)
        x = sum([f(x) for f in self.multi_preds])
        x = self.upscale(x)
        return x


class SalModel(nn.Module):
    def __init__(self, opt):
        self.name = 'SalSal_' + opt.base
        self.ws = 0.05
        net = DeepLab(pretrained=True, c_output=1, base=opt.base)
        net = torch.nn.parallel.DataParallel(net)
        self.net = net.cuda()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=opt.lr)

    def save(self, label):
        self.save_network(self.net, self.name, label)

    def load(self, label):
        print('loading %s'%label)
        self.load_network(self.net, self.name, label)

    def test(self, input, name, WW, HH):
        with torch.no_grad():
            big_mask_logits = self.net.forward(input.cuda())
            outputs = F.sigmoid(big_mask_logits.squeeze(1))
        outputs = outputs.detach().cpu().numpy() * 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((WW[ii], HH[ii]))
            msk.save('{}/{}.png'.format(self.opt.results_dir, name[ii]), 'PNG')

    def show_tensorboard(self, num_iter, num_show=4):
        loss = 0
        for k, v in self.loss.items():
            self.writer.add_scalar(k, v, num_iter)
            loss += v
        self.writer.add_scalar('total loss', loss, num_iter)
        num_show = min(self.input.size(0), num_show)


        img = self.input_sal.cpu()[:num_show]*self.v_std + self.v_mean
        self.writer.add_image('img_sal', torchvision.utils.make_grid(img), num_iter)

        pred_gt = self.targets_sal[:num_show]
        self.writer.add_image('gt_sal', torchvision.utils.make_grid(pred_gt.expand(-1, 3, -1, -1)).detach(), num_iter)

        pred = torch.sigmoid(self.big_mask_sal[:num_show])
        self.writer.add_image('prediction_sal', torchvision.utils.make_grid(pred.expand(-1, 3, -1, -1)).detach(), num_iter)


        img = self.input_syn.cpu()[:num_show]*self.v_std + self.v_mean
        self.writer.add_image('img_syn', torchvision.utils.make_grid(img), num_iter)

        pred_gt = self.targets_syn[:num_show]
        self.writer.add_image('gt_syn', torchvision.utils.make_grid(pred_gt.expand(-1, 3, -1, -1)).detach(), num_iter)

        pred = torch.sigmoid(self.big_mask_syn[:num_show])
        self.writer.add_image('prediction_syn', torchvision.utils.make_grid(pred.expand(-1, 3, -1, -1)).detach(), num_iter)



    def set_input(self, data):
        self.input_sal = data['img_sal'].cuda()
        self.targets_sal = data['gt_sal'].cuda()

        self.input_syn = data['img_syn'].cuda()
        self.targets_syn = data['gt_syn'].cuda()


    def sal_forward(self):
        # print("We are Forwarding !!")
            self.big_mask_sal = self.net.forward(self.input_sal)

    def syn_forward(self):
        # print("We are Forwarding !!")
            self.big_mask_syn = self.net.forward(self.input_syn)



    def sal_backward(self):
        # Combined loss
        loss = self.criterion(self.big_mask_sal, self.targets_sal) * (1-self.ws)
        gt_self = F.sigmoid(self.big_mask_sal).detach()
        gt_self[gt_self>0.5] = 1
        gt_self[gt_self<=0.5] = 0
        loss += self.criterion(self.big_mask_sal, gt_self) * self.ws
        loss.backward()
        self.loss['sal'] = loss.item()

    def syn_backward(self):
        # Combined loss
        loss = self.criterion(self.big_mask_syn, self.targets_syn)
        loss.backward()
        self.loss['syn'] = loss.item()

    def optimize_parameters(self, it):
        if it > 1000 and it % 500 == 0:
            self.optimizer.param_groups[0]['lr'] *= 0.5
        if it > 5000 and it % 500 == 0:
            self.optimizer.param_groups[0]['lr'] *= 0.2
        self.sal_forward()
        self.optimizer.zero_grad()
        self.sal_backward()
        self.optimizer.step()


        self.syn_forward()
        self.optimizer.zero_grad()
        self.syn_backward()
        self.optimizer.step()

    def switch_to_train(self):
        self.net.train()

    def switch_to_eval(self):
        self.net.eval()

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        save_filename = '_%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        device = next(network.parameters()).device
        torch.save(network.cpu().state_dict(), save_path)
        network.to(device)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '_%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        device = next(network.parameters()).device
        network.load_state_dict(torch.load(save_path, map_location={'cuda:%d' % device.index: 'cpu'}))



