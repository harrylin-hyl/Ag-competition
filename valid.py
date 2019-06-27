# python built-in library
import os
import argparse
import time
import csv
import uuid
# 3rd party library
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import cv2
# own code
from dataset import Compose
from helper import config, load_ckpt
import torch.nn.functional as F


class AgTestDataset(Dataset):
    """dataset."""

    def __init__(self, img_dir, transform=None, mode='test', split=False):
        """
        Args:
            root_dir (string): Directory of data (train or test).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        #throuth walk func get (parent,folders,files)
        files = next(os.walk(img_dir))[2]
        files.sort()
        self.mode = mode
        self.df = pd.DataFrame({'image_id': files, 'group': 0})
        if split == True:
            from copy import copy
            from sklearn.model_selection import train_test_split
            train, valid = train_test_split(
                self.df,
                test_size=config['dataset'].getfloat('cv_ratio'), 
                random_state=config['dataset'].getint('cv_seed'))
            if mode == 'train':
                self.df = train.reset_index(drop=True)
            else:
                self.df = valid.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            uid = self.df.loc[idx]['image_id']
        except:
            raise IndexError()
        image_path = os.path.join(self.img_dir,uid)
        image = Image.open(image_path)
        # ignore alpha channel if any, because they are constant in all training set
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # overlay masks to single mask
        w, h = image.size
        if self.mode == 'test':
            sample = {'image': image,
                        'uid': uid,
                        'size': image.size}
        else:
            print('mode erro!')
        if self.transform:
            sample = self.transform(sample)
        return sample


def main(ckpt, img_dir, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load one or more checkpoint
    models = []
    for fn in ckpt or [None]:
        # load model
        model = load_ckpt(filepath=fn)
        if not model:
            print("Aborted: checkpoint {} not found!".format(fn))
            return
        # Sets the model in evaluation mode.
        model.eval()
        # put model to GPU
        # Note: Comment out DataParallel due to
        #       (1) we won't need it in our inference scenario
        #       (2) it will change model's class name to 'dataparallel'
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)
        model = model.to(device)
        # append to model list
        models.append(model)

    resize = not config['valid'].getboolean('pred_orig_size')
    compose = Compose(augment=False, resize=resize)
    # decide which dataset to pick sample
    dataset = AgTestDataset(img_dir, transform=compose)

    # iterate dataset and inference each sample
    ious = []
    writer = csvfile = None
    for data in tqdm(dataset):
        with torch.no_grad():
            inference(data, models, resize, save_dir)
# end of main()

def inference(data, models, resize, save_dir):
    tta = config['valid'].getboolean('test_time_augment')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get input data
    uid = data['uid']
    size = data['size']
    inputs = data['image']
    # prepare input variables
    inputs = inputs.unsqueeze(0)
    inputs = inputs.to(device)

    if tta:
        txf_funcs = [lambda x: x,
                     lambda x: flip(x, 2), # up down flip
                     lambda x: flip(x, 3), # left right flip
                     lambda x: flip(flip(x, 3), 2),
                    ]
    else:
        txf_funcs = [lambda x: x]

    y_s = y_c = y_m = torch.FloatTensor().to(device)
    for model in models:
        model_name = type(model).__name__.lower()
        # predict model output
        for txf in txf_funcs:
            # apply test time transform 
            x = inputs
            x = txf(x)
            # padding
            if not resize:
                x = pad_tensor(x, size)
            # inference model
            s = model(x)
            pred_mask = F.softmax(s, 1).data.max(1)[1].squeeze().cpu().numpy()
    p = Image.fromarray(pred_mask.astype(np.uint8))
    png_uid = uid.split('.')[0] + '.png'
    p.save(save_dir + '/' + png_uid)
# end of predict()

def flip(t, dim):
    dim = t.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
            else t.new(torch.arange(t.size(i)-1, -1, -1).tolist()).long()
            for i in range(t.dim()))
    return t[inds]


def pad_tensor(img_tensor, size, mode='reflect'):
    # get proper mini-width required for model input
    # for example, 32 for 5 layers of max_pool
    gcd = config['param'].getint('gcd_depth')
    # estimate border padding margin
    # (paddingLeft, paddingRight, paddingTop, paddingBottom)
    pad_w = pad_h = 0
    w, h = size
    if 0 != (w % gcd):
        pad_w = gcd - (w % gcd)
    if 0 != (h % gcd):
        pad_h = gcd - (h % gcd)
    pad = (0, pad_w, 0, pad_h)
    # decide padding mode
    if mode == 'replica':
        f = nn.ReplicationPad2d(pad)
    elif mode == 'constant':
        # padding color should honor each image background, default is black (0)
        bgcolor = 0 if np.median(img_tensor) < 100 else 255
        f = nn.ConstantPad2d(pad, bgcolor)
    elif mode == 'reflect':
        f = nn.ReflectionPad2d(pad)
    else:
        raise NotImplementedError()
    return f(img_tensor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', help='Specify dataset to evaluate')
    parser.add_argument('--save_dir', help='Save overlay prediction as PNG files')
    parser.add_argument('ckpt', nargs='*', help='filepath of checkpoint(s), otherwise lookup checkpoint/current.json')
    parser.set_defaults(img_dir='./round1_test/image_4', save_dir='./output/image_4')
    args = parser.parse_args()

    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)

    main(args.ckpt, args.img_dir, args.save_dir)
