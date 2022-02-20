import argparse

import numpy as np
import torch
from fastcore.all import Path

import pandas as pd
from fastcore.foundation import L
from sklearn.metrics import f1_score, cohen_kappa_score
from torch.utils.data import WeightedRandomSampler, DataLoader
from tqdm import tqdm

from config import *
from draw import show_images
from model import ResNet18
from augs import train_augs, valid_augs
from utils import APTOSDataset, update_model_wt


def make_ds(train_data, valid_data, train_pth, valid_pth, train_augs=None, valid_augs=None, resize_sz=384):
    """Prepare torch dataset"""
    train_ds = APTOSDataset(train_data, pth=train_pth, augs=train_augs, sz=resize_sz)
    valid_ds = APTOSDataset(valid_data, pth=valid_pth, augs=valid_augs, sz=resize_sz)
    return train_ds, valid_ds


def loss_func(acts, labels):
    return torch.nn.CrossEntropyLoss()(acts, labels.squeeze())


def get_f1score(y_true, y_pred):
    return f1_score(y_true.squeeze().detach().cpu(), y_pred.squeeze().detach().cpu(), average='micro')


def get_ckscore(y_true, y_pred):
    return cohen_kappa_score(y_true.squeeze().detach().cpu(), y_pred.squeeze().detach().cpu(), weights='quadratic')


def show_preds(dl, limit=6, **kwargs):
    """Method to show predictions for a batch from dataloader."""
    model.eval()
    images, labels = next(iter(dl))
    images, labels = images.to(device), labels.to(device)
    acts = model(images)
    preds = torch.softmax(acts, -1).argmax(-1, keepdim=True)
    show_images(images[:limit], labels[:limit], preds[:limit], **kwargs)


@torch.no_grad()
def validate_epoch(valid_dl, model, loss_func=loss_func, show=False):
    model.eval()
    l_valid = len(valid_dl)
    valid_loss = 0.0
    valid_ck = 0.0
    valid_f1 = 0.0
    for vb, (images, labels) in enumerate(tqdm(valid_dl)):
        images, labels = images.to(device), labels.to(device)
        acts = model(images)
        loss = loss_func(acts, labels.squeeze())
        valid_loss += loss.item()
        preds = torch.softmax(acts, -1).argmax(-1, keepdim=True)
        valid_ck += get_ckscore(labels, preds)
        valid_f1 += get_f1score(labels, preds)
    tqdm.write(
        f'valid_score:{(valid_ck / l_valid):.4f} valid_f1:{(valid_f1 / l_valid):.4f} valid_loss:{(valid_loss / l_valid):.4f}')
    if show:
        show_preds(valid_dl, 3)


def train(train_dl, valid_dl, epochs):
    for e in range(epochs):
        print(f'epoch {e}')
        model.train()
        train_loss = 0.0
        train_ck = 0.0
        train_f1 = 0.0
        l_train = len(train_dl)
        for tb, (images, labels) in enumerate(tqdm(train_dl)):
            images, labels = images.to(device), labels.to(device)
            acts = model(images)
            # opt
            optim.zero_grad()
            # loss
            loss = loss_func(acts, labels.squeeze())
            train_loss += loss.item()
            loss.backward()
            optim.step()
            # preds train
            preds = torch.softmax(acts, -1).argmax(-1, keepdim=True)
            train_ck += get_ckscore(labels, preds)
            train_f1 += get_f1score(labels, preds)
        tqdm.write(
            f'train_score:{(train_ck / l_train):.4f} train_f1:{(train_f1 / l_train):.4f} train_loss:{(train_loss / l_train):.4f}')
        if e % 2 == 0:
            # validate every 2 epochs
            validate_epoch(valid_dl, model, loss_func=loss_func, show=show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for retinopathy classification')
    parser.add_argument('fepochs', metavar='fe', type=int,
                        help='number of epochs to train with frozen body')
    parser.add_argument('uepochs', metavar='ue', type=int,
                        help='number of epochs to train the full model')
    parser.add_argument('--csv', metavar='d', type=str,
                        help='path to csv file containing image id_code and diagnosis', default=Path('data/train.csv'))
    parser.add_argument('--data', metavar='d', type=str,
                        help='path to find training data', default=Path('data/train_images'))
    parser.add_argument('--ext', metavar='ext', type=str,
                        help='image file extension', default='.png')
    args = parser.parse_args()

    # print args
    print(f'frozen epochs: {args.fepochs}')
    print(f'unfrozen epochs: {args.uepochs}')
    print(f'data directory: {args.data}')
    print(f'file extension: {args.ext}')
    print(f'train csv: {args.csv}')

    # begin
    train_df = pd.read_csv(args.csv)
    counts = train_df.diagnosis.value_counts()
    weights = counts / counts.sum()
    train_df["weights"] = train_df.diagnosis.apply(lambda x: -np.log(weights[x]))

    train_pth = Path(args.data)
    pths = train_pth.ls(file_exts=args.ext)

    print(f'training image files: {len(pths)}')
    print(f'vocab: {vocab}')

    data = train_df.sample(frac=1, replace=False).to_records(index=False)
    split_sz = round(len(data) * 0.8)
    train_data = data[:split_sz]
    valid_data = data[split_sz:]

    print(f'train size: {len(train_data)}')
    print(f'valid size: {len(valid_data)}')

    train_ds, valid_ds = make_ds(train_data, valid_data, train_pth, train_pth,
                                 train_augs=train_augs, valid_augs=valid_augs,
                                 resize_sz=resize_sz)

    train_sampler = WeightedRandomSampler(weights=train_data.weights, num_samples=len(train_ds), replacement=False)
    valid_sampler = WeightedRandomSampler(weights=valid_data.weights, num_samples=len(valid_ds), replacement=False)
    train_dl = DataLoader(train_ds, batch_size=bs, drop_last=drop_last, sampler=train_sampler)
    valid_dl = DataLoader(valid_ds, batch_size=bs, drop_last=drop_last, sampler=valid_sampler)

    print(f'train dataloader: {len(train_dl)}')
    print(f'valid dataloader: {len(valid_dl)}')
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    model = ResNet18(n_cls)
    if use_imagenet_wts:
        model = update_model_wt(model)
    model = model.to(device)

    # freeze
    for name, p in model.named_parameters():
        if 'fc' not in name:
            p.requires_grad_(False)

    optim = torch.optim.Adam(model.fc.parameters(), lr=lr_head_f)  # only train the head
    print('training with frozen params')
    train(train_dl, valid_dl, args.fepochs)
    torch.save({
        'epoch': args.fepochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict()
    }, f'checkpoints/model_c{args.fepochs}.pth')

    optim.add_param_group({'params': L(model.parameters())[:-2], 'lr': lr_body_uf})  # make body trainable
    optim.param_groups[0]['lr'] = lr_head_uf
    
    # unfreeze
    for name, p in model.named_parameters():
        p.requires_grad_(True)

    print('training full model')
    train(train_dl, valid_dl, args.uepochs)
    torch.save({
        'epoch': args.fepochs + args.uepochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict()
    }, f'checkpoints/model_c{args.fepochs + args.uepochs}.pth')

    print(f'training complete for {args.fepochs + args.uepochs} epochs')
    print(f'checkpoints saved to checkpoints/model_c*.pth')
    
