import os
import random
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tx

from PIL import Image, ImageOps, ImageDraw
from skimage.io import imread

# Ignore skimage convertion warnings
import warnings
warnings.filterwarnings("ignore")

from helper import config, clahe


class AgDataset(Dataset):
    """dataset."""

    def __init__(self, root, transform=None, mode='train', split=True):
        """
        Args:
            root_dir (string): Directory of data (train or test).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        #throuth walk func get (parent,folders,files)
        files = next(os.walk(os.path.join(root,'image')))[2]
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
        image_path = os.path.join(self.root,'image',uid)
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
            label_path = os.path.join(self.root, 'label',uid)
            #load masks from json label file
            label = Image.open(label_path)
            # if label.mode != 'L':
            #     label = label.convert('L')
            
            sample = {'image': image,
                    'label': label,
                    'uid': uid,
                    'size': image.size}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def split(self):
        ''' return CV split dataset object '''
        from copy import copy
        from sklearn.model_selection import train_test_split
        train, valid = train_test_split(
            self.df,
            test_size=config['dataset'].getfloat('cv_ratio'), 
            random_state=config['dataset'].getint('cv_seed'))
        train_dataset = copy(self)
        valid_dataset = copy(self)
        train_dataset.df = train.reset_index(drop=True)
        valid_dataset.df = valid.reset_index(drop=True)
        return train_dataset, valid_dataset


class Compose():
    def __init__(self, augment=True, resize=False, tensor=True):
        self.tensor = tensor
        self.augment = augment
        self.resize = resize

        model_name = config['param']['model']
        width = config.getint(model_name, 'width')
        self.size = (width, width)
        self.weight_map = config['param'].getboolean('weight_map')

        c = config['pre']
        self.mean = json.loads(c.get('mean'))
        self.std = json.loads(c.get('std'))
        self.label_binary = c.getboolean('label_to_binary')
        self.color_invert = c.getboolean('color_invert')
        self.color_jitter = c.getboolean('color_jitter')
        self.color_equalize = c.getboolean('color_equalize')
        self.min_scale = c.getfloat('min_scale')
        self.max_scale = c.getfloat('max_scale')
        self.add_noise = c.getboolean('add_noise')
        self.channel_shuffle = c.getboolean('channel_shuffle')

    def __call__(self, sample):
        if not 'label' in sample.keys():
            image = sample['image']
            if self.resize:  # resize down image
                image= tx.resize(image, self.size)
            # perform ToTensor()
            if self.tensor:
                image = tx.to_tensor(image)
                # perform Normalize()
                image = tx.normalize(image, self.mean, self.std)

            # prepare a shadow copy of composed data to avoid screwup cached data
            x = sample.copy()
            x['image'] = image
            return x

        image, label = sample['image'], sample['label']

        if self.augment:
            if self.color_equalize and random.random() > 0.5:
                image = clahe(image)

            # perform RandomResize() or just enlarge for image size < model input size
            if random.random() > 0.5:
                new_size = int(random.uniform(self.min_scale, self.max_scale) * np.min(image.size))
            else:
                new_size = int(np.min(image.size))
            if new_size < np.max(self.size): # make it viable for cropping
                new_size = int(np.max(self.size))
            image, label = [tx.resize(x, new_size) for x in (image, label)]

            # perform RandomCrop()
            i, j, h, w = transforms.RandomCrop.get_params(image, self.size)
            image, label = [tx.crop(x, i, j, h, w) for x in (image, label)]

            # perform RandomHorizontalFlip()
            if random.random() > 0.5:
                image, label = [tx.hflip(x) for x in (image, label)]

            # perform RandomVerticalFlip()
            if random.random() > 0.5:
                image, label = [tx.vflip(x) for x in (image, label)]

            # perform Random Rotation (0, 90, 180, and 270 degrees)
            random_degree = random.randint(0, 3) * 90
            image, label = [tx.rotate(x, random_degree) for x in (image, label)]

            # perform channel shuffle
            if self.channel_shuffle:
                image = ChannelShuffle()(image)

            # perform random color invert, assuming 3 channels (rgb) images
            if self.color_invert and random.random() > 0.5:
                image = ImageOps.invert(image)

            # # perform ColorJitter()
            # if self.color_jitter and random.random() > 0.5:
            #     color = transforms.ColorJitter.get_params(0.5, 0.5, 0.5, 0.25)
            #     image = color(image)

            if self.add_noise and random.random() > 0.5:
                image = add_noise(image)
            
        elif self.resize:  # resize down image
            image, label = [tx.resize(x, self.size) for x in (image, label)]
        # perform ToTensor()
        if self.tensor:
            # image, label = \
            #         [tx.to_tensor(x) for x in (image, label)]
            image = tx.to_tensor(image)
            label = torch.tensor(np.array(label))
            # perform Normalize()
            image = tx.normalize(image, self.mean, self.std)

        # prepare a shadow copy of composed data to avoid screwup cached data
        x = sample.copy()
        x['image'], x['label'] = image, label

        return x

    def denorm(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

    def pil(self, tensor):
        return tx.to_pil_image(tensor)

    def to_numpy(self, tensor, size=None):
        x = tx.to_pil_image(tensor)
        if size is not None:
            x = x.resize(size)
        return np.asarray(x)

    def show(self, sample):
        image, label, label_c, label_m, label_gt = \
                sample['image'], sample['label'], sample['label_c'], sample['label_m'], sample['label_gt']
        for x in (image, label, label_c, label_m, label_gt):
            if x.dim == 4:  # only dislay first sample
                x = x[0]
            if x.shape[0] > 1: # channel > 1
                x = self.denorm(x)
            x = self.pil(x)
            x.show()

def compose_mask(masks, pil=False):
    # result = np.zeros_like(masks[0], dtype=np.int32)
    result = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), dtype=np.int32)
    b_i = 0
    r_i = 0
    for i, m in enumerate(masks):
        mask = np.array(m) if pil else m.copy()
        if mask.max() == 1:
            result[:,:,0][mask == 1] = b_i + 1
            b_i += 1
        else:
            result[:,:,1][mask == 2] = r_i + 1
            r_i += 1
        # mask[mask > 0] = i + 1 # zero for background, starting from 1
        # result = np.maximum(result, mask) # overlay mask one by one via np.maximum, to handle overlapped labels if any
    if pil:
        result = Image.fromarray(result)
    return result

def decompose_mask(mask):
    num = mask.max()
    result = []
    for i in range(1, num+1):
        m = mask.copy()
        m[m != i] = 0
        m[m == i] = 255
        result.append(m)
    return result

def add_noise(x, mode='speckle'):
    from skimage.util import random_noise
    is_pil = isinstance(x, Image.Image)
    if is_pil:
        x = np.asarray(x, dtype=np.uint8)
    # input numpy array, and return [0, 1] or [-1, 1] array
    x = random_noise(x, mode=mode)
    if is_pil:
        x = (x * 255).astype(np.uint8)
        x = Image.fromarray(x)
    return x

class ChannelShuffle():
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Shuffled channel image.
        """
        assert isinstance(img, Image.Image)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if random.random() < 0.3:
            # shuffle to grayscale
            img = img.convert('L')
            img = img.convert('RGB')
        else:
            r, g, b = img.split()
            c = [r, g, b]
            np.random.shuffle(c)
            img = Image.merge("RGB", c)
        return img

if __name__ == '__main__':
    compose = Compose(augment=False)
    train = AgDataset('../../Data/round1_train', transform=compose,mode='valid')
    print(len(train))
    sample = train[0]
    print(sample['uid'])
    print(sample['image'])
    print(sample['label'])


