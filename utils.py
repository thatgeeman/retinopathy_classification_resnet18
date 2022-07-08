import os

import numpy as np
import torch
from PIL import Image
from fastcore.all import Path
from torch import nn
from torch.utils.data import Dataset
from torchvision import models

from config import stats


def open_img(img_id, pth: Path, ext='png', cspace='RGB', sz=None, pil=False):
    """Read image from path given the image_id."""
    img_pth = pth / f'{img_id}.{ext}'
    img = Image.open(img_pth).convert(cspace)
    img = img if sz is None else img.resize((sz, sz))
    return np.asarray(img) if not pil else img


def collate_fn_test(b):
    """Collate data for test dataloader."""
    batch = list(zip(*b))
    images = torch.stack(batch[0], 0)
    labels = None
    return images, labels


class APTOSDataset(Dataset):
    def __init__(self, data, pth, augs=None, sz=300, test=False):
        super(APTOSDataset, self).__init__()
        self.data = data
        self.pth = pth
        self.augs = augs(sz, stats["mean"], stats["std"]) if augs is not None else False
        self.sz = sz
        self.test = test

    def __getitem__(self, idx):
        label = None
        img = open_img(self.data[idx][0], self.pth, sz=self.sz, pil=True)
        if not self.test:
            label = self.data[idx][1]
            label = torch.tensor([label], dtype=torch.long)
        if self.augs:
            img = self.augs(img)
        return img, label

    def __len__(self):
        return len(self.data)


def update_model_wt(model):
    """Update model with imagenet weights from torchvision"""
    torch_model = models.resnet18(pretrained=True)
    with torch.no_grad():
        model.l1[0].conv.weight = nn.Parameter(torch_model.conv1.weight)  # conv1 weight
        model.l1[0].bn.weight = nn.Parameter(torch_model.bn1.weight)  # bn1 weight
        model.l2.reg_blk0.conv1.conv.weight = nn.Parameter(torch_model.layer1[0].conv1.weight)
        model.l2.reg_blk0.conv1.bn.weight = nn.Parameter(torch_model.layer1[0].bn1.weight)
        model.l2.reg_blk0.conv2.conv.weight = nn.Parameter(torch_model.layer1[0].conv2.weight)
        model.l2.reg_blk0.conv2.bn.weight = nn.Parameter(torch_model.layer1[0].bn2.weight)
        #
        model.l2.reg_blk1.conv1.conv.weight = nn.Parameter(torch_model.layer1[1].conv1.weight)
        model.l2.reg_blk1.conv1.bn.weight = nn.Parameter(torch_model.layer1[1].bn1.weight)
        model.l2.reg_blk1.conv2.conv.weight = nn.Parameter(torch_model.layer1[1].conv2.weight)
        model.l2.reg_blk1.conv2.bn.weight = nn.Parameter(torch_model.layer1[1].bn2.weight)
        model.l3.conv_blk0.conv1.conv.weight = nn.Parameter(torch_model.layer2[0].conv1.weight)
        model.l3.conv_blk0.conv1.bn.weight = nn.Parameter(torch_model.layer2[0].bn1.weight)
        model.l3.conv_blk0.conv2.conv.weight = nn.Parameter(torch_model.layer2[0].conv2.weight)
        model.l3.conv_blk0.conv2.bn.weight = nn.Parameter(torch_model.layer2[0].bn2.weight)
        model.l3.conv_blk0.dsample.conv.weight = nn.Parameter(torch_model.layer2[0].downsample[0].weight)
        model.l3.conv_blk0.dsample.bn.weight = nn.Parameter(torch_model.layer2[0].downsample[1].weight)
        #
        model.l3.reg_blk1.conv1.conv.weight = nn.Parameter(torch_model.layer2[1].conv1.weight)
        model.l3.reg_blk1.conv1.bn.weight = nn.Parameter(torch_model.layer2[1].bn1.weight)
        model.l3.reg_blk1.conv2.conv.weight = nn.Parameter(torch_model.layer2[1].conv2.weight)
        model.l3.reg_blk1.conv2.bn.weight = nn.Parameter(torch_model.layer2[1].bn2.weight)
        model.l4.conv_blk0.conv1.conv.weight = nn.Parameter(torch_model.layer3[0].conv1.weight)
        model.l4.conv_blk0.conv1.bn.weight = nn.Parameter(torch_model.layer3[0].bn1.weight)
        model.l4.conv_blk0.conv2.conv.weight = nn.Parameter(torch_model.layer3[0].conv2.weight)
        model.l4.conv_blk0.conv2.bn.weight = nn.Parameter(torch_model.layer3[0].bn2.weight)
        model.l4.conv_blk0.dsample.conv.weight = nn.Parameter(torch_model.layer3[0].downsample[0].weight)
        model.l4.conv_blk0.dsample.bn.weight = nn.Parameter(torch_model.layer3[0].downsample[1].weight)
        #
        model.l4.reg_blk1.conv1.conv.weight = nn.Parameter(torch_model.layer3[1].conv1.weight)
        model.l4.reg_blk1.conv1.bn.weight = nn.Parameter(torch_model.layer3[1].bn1.weight)
        model.l4.reg_blk1.conv2.conv.weight = nn.Parameter(torch_model.layer3[1].conv2.weight)
        model.l4.reg_blk1.conv2.bn.weight = nn.Parameter(torch_model.layer3[1].bn2.weight)
        model.l5.conv_blk0.conv1.conv.weight = nn.Parameter(torch_model.layer4[0].conv1.weight)
        model.l5.conv_blk0.conv1.bn.weight = nn.Parameter(torch_model.layer4[0].bn1.weight)
        model.l5.conv_blk0.conv2.conv.weight = nn.Parameter(torch_model.layer4[0].conv2.weight)
        model.l5.conv_blk0.conv2.bn.weight = nn.Parameter(torch_model.layer4[0].bn2.weight)
        model.l5.conv_blk0.dsample.conv.weight = nn.Parameter(torch_model.layer4[0].downsample[0].weight)
        model.l5.conv_blk0.dsample.bn.weight = nn.Parameter(torch_model.layer4[0].downsample[1].weight)
        #
        model.l5.reg_blk1.conv1.conv.weight = nn.Parameter(torch_model.layer4[1].conv1.weight)
        model.l5.reg_blk1.conv1.bn.weight = nn.Parameter(torch_model.layer4[1].bn1.weight)
        model.l5.reg_blk1.conv2.conv.weight = nn.Parameter(torch_model.layer4[1].conv2.weight)
        model.l5.reg_blk1.conv2.bn.weight = nn.Parameter(torch_model.layer4[1].bn2.weight)
    return model


def set_seed(seed=42):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed()