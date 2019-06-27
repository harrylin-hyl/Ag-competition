# python built-in library
import os
import argparse
import time
from multiprocessing import Manager
# 3rd party library
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from tensorboardX import SummaryWriter
from torchvision import utils
# own code
from model import build_model
from dataset import AgDataset, Compose
from helper import config, AverageMeter, iou_mean, save_ckpt, load_ckpt
from loss import softmax_focal_criterion
from lr import CyclicLR
import cv2
import torch.nn.functional as F
from encoding.models.danet import DANet


def main(resume=True, n_epoch=None, learn_rate=None, ckpt_name='checkpoint'):
    model_name = config['param']['model']
    if learn_rate is None:
        learn_rate = config['param'].getfloat('learn_rate')
    width = config.getint(model_name, 'width')
    weight_map = config['param'].getboolean('weight_map')
    c = config['train']
    log_name = c.get('log_name')
    n_batch = c.getint('n_batch')
    n_worker = c.getint('n_worker')
    n_cv_epoch = c.getint('n_cv_epoch')
    if n_epoch is None:
        n_epoch = c.getint('n_epoch')
    balance_group = c.getboolean('balance_group')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = DANet(4,
                backbone='resnet101',
                aux=False, se_loss=False,
                base_size=512, crop_size=512,
                multi_grid=True,
                multi_dilation=[4,8,16])

    # model = build_model(model_name)
    model = model.to(device)

    # dataloader workers are forked process thus we need a IPC manager to keep cache in same memory space
    train_compose = Compose()
    val_compose = Compose(augment=False)
    # prepare dataset
    train_dataset = AgDataset('./round1_train', transform=train_compose,mode='train')
    valid_dataset = AgDataset('./round1_train', transform=train_compose,mode='valid')
    # train_dataset = AgDataset('./data-test', transform=train_compose,mode='train')
    # valid_dataset = AgDataset('./data-test', transform=train_compose,mode='valid')

    sampler = RandomSampler(train_dataset)
    
    # data loader
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=n_batch,
        num_workers=n_worker,
        pin_memory=torch.cuda.is_available(),
        drop_last=True)
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=n_worker,
        drop_last=True)

    # warm-up
    # lrer = CyclicLR(3, 300, 0.5, len(train_loader), 1, 1e-14)
    lrer = None
    if lrer == None:
        # # define optimizer
        # optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, model.parameters()),
        #     lr=args.learn_rate,
        #     weight_decay=1e-6
        #     )
        ##DAnet optimizer
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.learn_rate},]
        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'lr': args.learn_rate*10})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'lr': args.learn_rate*10})
        optimizer = torch.optim.SGD(params_list,
                    lr=args.learn_rate,
                    momentum=0.9,
                    weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-14,
            momentum=0.9
            # lr=args.lr_base
            )
    # resume checkpoint
    start_epoch = iou_tr = iou_cv = 0
    if resume:
        start_epoch = load_ckpt(model, optimizer, os.path.join('.', ckpt_name))
    if start_epoch == 0:
        print('Grand new training ...')

    # put model to GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # decide log directory name
    log_dir = os.path.join(
        'logs', log_name, '{}-{}'.format(model_name, width),
        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
        )

    # with SummaryWriter(log_dir) as writer:
    writer = SummaryWriter(log_dir)
    if start_epoch == 0 and False:
        # dump graph only for very first training, disable by default
        dump_graph(model, writer, n_batch, width)
    print('Training started...')
    for epoch in range(start_epoch + 1, n_epoch + start_epoch + 1): # 1 base
        loss_tr = train(train_loader, model, optimizer, epoch, writer, lrer)
        if len(valid_dataset) > 0 and epoch % n_cv_epoch == 0:
            with torch.no_grad():
                iou_cv = valid(valid_loader, model, epoch, writer, len(train_loader))
        save_ckpt(model, optimizer, epoch, iou_cv, ckpt_name)
    writer.close()
    print('Training finished...')

def dump_graph(model, writer, n_batch, width):
    # Prerequisite
    # $ sudo apt-get install libprotobuf-dev protobuf-compiler
    # $ pip3 install onnx
    print('Dump model graph...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.rand(n_batch, 3, width, width, device=device)
    torch.onnx.export(model, dummy_input, "checkpoint/model.pb", verbose=False)
    writer.add_graph_onnx("checkpoint/model.pb")

def train(loader, model, optimizer, epoch, writer, lrer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    iou = AverageMeter()   # semantic IoU
    print_freq = config['train'].getfloat('print_freq')
    model_name = config['param']['model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sets the module in training mode.
    model.train()
    end = time.time()
    n_step = len(loader)
    for i, data in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # split sample data
        inputs = data['image'].to(device)
        labels = data['label'].to(device)

        # warm-up-lr
        if lrer != None:
            lr = lrer.lr
            CyclicLR.update_lr(lr, optimizer)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward step
        outputs = model(inputs)
        loss = softmax_focal_criterion(outputs[0], labels)
        loss += softmax_focal_criterion(outputs[1], labels)
        loss += softmax_focal_criterion(outputs[2], labels)
        # compute gradient and do backward step
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # measure accuracy and record loss
        # NOT instance-level IoU in training phase, for better speed & instance separation handled in post-processing
        losses.update(loss.item(), inputs.size(0))

        if (i + 1) % print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time: {batch_time.avg:.2f} (io: {data_time.avg:.2f})\t'
                'Loss: {loss.val:.4f} (avg: {loss.avg:.4f})\t'
                .format(
                    epoch, i, n_step, batch_time=batch_time,
                    data_time=data_time, loss=losses
                )
            )
    # end of loop, dump epoch summary
    writer.add_scalar('training/epoch_loss', losses.avg, epoch)
    return losses.avg # return epoch average iou

def valid(loader, model, epoch, writer, n_step):
    iou = AverageMeter()   # semantic IoU
    iou_1 = AverageMeter()
    iou_2 = AverageMeter()
    iou_3 = AverageMeter()
    losses = AverageMeter()
    model_name = config['param']['model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sets the model in evaluation mode.
    model.eval()
    n_step = len(loader)
    for i, data in enumerate(loader):
        # get the inputs
        inputs = data['image'].to(device)
        labels = data['label'].to(device)
        # forward step
        outputs = model(inputs)
        # compute loss

        loss = softmax_focal_criterion(outputs, labels)

        # measure accuracy and record loss (Non-instance level IoU)
        losses.update(loss.item(), inputs.size(0))

        # semantic label output
        pred_mask = F.softmax(outputs, 1).data.max(1)[1].squeeze().cpu().numpy()
        targets = labels.to('cpu').detach().numpy()
        # compute iou 
        batch_iou_1 = iou_mean(pred_mask==1, targets==1)
        batch_iou_2 = iou_mean(pred_mask==2, targets==2)
        batch_iou_3 = iou_mean(pred_mask==3, targets==3)
        batch_iou = (batch_iou_1 + batch_iou_2 + batch_iou_3)/3
        iou.update(batch_iou, inputs.size(0))
        iou_1.update(batch_iou_1, inputs.size(0))
        iou_2.update(batch_iou_2, inputs.size(0))
        iou_3.update(batch_iou_3, inputs.size(0))
        # add image to writer, only first batch
        # outputs = F.softmax(outputs)
        # image_list = [(labels[0][0]==1)[None].float(), outputs[0,1,:,:][None], (labels[0][0]==2)[None].float(), outputs[0,2,:,:][None],
        #                 (labels[0][0]==3)[None].float(), outputs[0,3,:,:][None], (labels[0][0]==4)[None].float(), outputs[0,4,:,:][None], 
        #                 labels[0][1][None], outputs[0,5,:,:][None] ]
        # debug = utils.make_grid(image_list, nrow=2)
        # writer.add_image('images/debug', debug, i + epoch * n_step)
        # writer.add_image('images/b_label', labels[0][0], i + epoch * n_step)
        # writer.add_image('images/r_label', labels[0][1], i + epoch * n_step)
        # writer.add_image('images/b_output', outputs[0][0], i + epoch * n_step)
        # writer.add_image('images/r_output', outputs[0][1], i + epoch * n_step)
        # writer.add_image('images/c_label', labels_c[0], i + epoch * n_step)
        # writer.add_image('images/c_output', outputs_c[0], i + epoch * n_step)
    # end of loop, dump epoch summary
    writer.add_scalar('CV/epoch_loss', losses.avg, epoch)
    writer.add_scalar('CV/epoch_iou', iou.avg, epoch)
    writer.add_scalar('CV/epoch_iou_1', iou_1.avg, epoch)
    writer.add_scalar('CV/epoch_iou_2', iou_2.avg, epoch)
    writer.add_scalar('CV/epoch_iou_3', iou_3.avg, epoch)
    print(
        'Epoch: [{0}]\t\tcross-validation\t'
        'Loss: N/A    (avg: {loss.avg:.4f})\t'
        'IoU: {iou.avg:.3f}\t'
        .format(
            epoch, loss=losses, iou=iou
        )
    )
    return iou.avg # return epoch average iou

if __name__ == '__main__':
    learn_rate = config['param'].getfloat('learn_rate')
    n_epoch = config['train'].getint('n_epoch')
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_name', default='checkpoint', help='name of logs to save')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--epoch', type=int, help='run number of epoch')
    parser.add_argument('--lr', type=float, dest='learn_rate', help='learning rate')
    parser.set_defaults(resume=True, epoch=n_epoch, learn_rate=learn_rate)
    args = parser.parse_args()
    if not os.path.exists(os.path.join('.', args.ckpt_name)):
        os.mkdir(os.path.join('.', args.ckpt_name))

    main(args.resume, args.epoch, args.learn_rate, args.ckpt_name)
