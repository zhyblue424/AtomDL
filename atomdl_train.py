# unet.py
#

from __future__ import division

from math import sqrt
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from sfin import SFIN
import numpy as np
from scipy import ndimage
from utils import *
import argparse, os, time, random
import matplotlib.pyplot as plt
import numpy as np
def parse_option():
    parser = argparse.ArgumentParser('AtomSegementation Training', add_help=False)
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--val_split', type=float, default=0.2, help='validation split ratio')#0.1
    parser.add_argument('--test_split', type=float, default=0.0, help='testing split ratio')#0.1
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--gpu', type=int, default=1, help='gpu device id')

    """New arguments for paths"""
    parser.add_argument('--root_inputs', type=str, default='./datasets', help='Root directory for inputs and labels')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_option()
    CUDA_VISIBLE_DEVICES=args.gpu
    images_dir=args.root + '/image'
    labels_dir=args.root + '/circularMask'
    train_loader, val_loader = get_dataloaders(images_dir, labels_dir, batch_size = args.batch_size, val_split=args.val_split, test_split=args.test_split, num_workers=args.num_workers)

    print('Loading model...')
    unet = SFIN().cuda()
    num_total = sum(p.numel() for p in unet.parameters())
    print(f"Total parameters: {num_total:,}")
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [250, 400, 450, 475], gamma=0.5)

    print('Start training...')
    batch_losses = []
    batch_mious = []
    for epoch in tqdm(range(args.epochs)):
        unet.train()
        l1_loss = nn.L1Loss(reduction='mean')
        running_loss = 0.0
        num_bathces = 0

        train_total_miou = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch[{epoch:.0f}] Processing Training Batches", unit='batch')
        for inputs, labels in train_loader_tqdm:      
            optimizer.zero_grad()
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = unet(inputs)
            loss = l1_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

            outputs_bin = torch.clip(outputs, 0, 255)
            outputs_bin = (outputs > 128).float()
            labels = (labels > 128).float()
            
            _, _, _, train_miou = compute_metric(outputs_bin, labels)
            batch_mious.append(train_miou.item())

            train_loader_tqdm.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'mIoU': f'{train_miou.item():.4f}'
            })
            

        scheduler.step()  
        epoch_model_path = os.path.join(args.checkpoint_dir, "SFIN_TEM_ImageNet.pth")
        torch.save(unet.state_dict(), epoch_model_path)
        print(f"\n\n Model saved to {epoch_model_path}")
        np.save(os.path.join(args.output_dir, 'batch_losses.csv'), np.array(batch_losses))
        np.save(os.path.join(args.output_dir, 'batch_mious.csv'), np.array(batch_mious))


    print('Evaluating on validation set...')

    unet.eval()
    val_total_miou, val_total_recall, val_total_precision, val_total_f1 = 0.0, 0.0, 0.0, 0.0
    val_loader_tqdm = tqdm(val_loader, desc='Processing Validation Batches', unit='batch')
    for inputs, labels in val_loader_tqdm:      
        inputs, labels = inputs.cuda(0), labels.cuda(0)
        outputs = unet(inputs)
        outputs = torch.clip_(outputs, 0, 255)
        outputs = (outputs > 128).float()
        labels = (labels > 128).float()
        recall, precision, f1, iou = compute_metric(outputs, labels) 
        val_total_miou += iou * inputs.size(0)
        val_total_recall += recall * inputs.size(0)
        val_total_precision += precision * inputs.size(0)
        val_total_f1 += f1 * inputs.size(0)
    val_avg_miou = val_total_miou / len(val_loader.dataset)
    val_avg_recall = val_total_recall / len(val_loader.dataset)
    val_avg_precision = val_total_precision / len(val_loader.dataset)
    val_avg_f1 = val_total_f1 / len(val_loader.dataset)
    print(f"Validation Recall: {val_avg_recall:.4f}, Validation Precision: {val_avg_precision:.4f}, Validation F1 Score: {val_avg_f1:.4f}, Validation mIoU: {val_avg_miou:.4f}")
