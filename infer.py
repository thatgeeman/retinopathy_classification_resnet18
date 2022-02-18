import argparse

import torch
from fastcore.all import Path

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from augs import valid_augs
from config import *
from model import ResNet18
from utils import APTOSDataset, collate_fn_test


@torch.no_grad()
def infer(test_dl, model):
    model.eval()
    result = []
    for vb, (images, labels) in enumerate(tqdm(test_dl)):
        images = images.to(device)
        acts = model(images)
        preds = torch.softmax(acts, -1).argmax(-1, keepdim=True)
        result.extend(preds.squeeze().tolist())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script for retinopathy classification')
    parser.add_argument('checkpoint', metavar='model', type=str,
                        help='path to find model checkpoint', default=Path('checkpoints/model_c15.pth'))
    parser.add_argument('--csv', metavar='d', type=str,
                        help='path to csv file containing image id_code and diagnosis', default=Path('data/test.csv'))
    parser.add_argument('--data', metavar='d', type=str,
                        help='path to find training data', default=Path('data/test_images'))
    parser.add_argument('--ext', metavar='ext', type=str,
                        help='image file extension', default='.png')
    args = parser.parse_args()

    # print args
    print(f'model checkpoint: {args.checkpoint}')
    print(f'data directory: {args.data}')
    print(f'file extension: {args.ext}')
    print(f'train csv: {args.csv}')

    test_pth = Path(args.data)
    test_df = pd.read_csv(args.csv)
    test_data = test_df.to_records(index=False)
    test_ds = APTOSDataset(test_data, pth=test_pth, augs=valid_augs, sz=resize_sz, test=True)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False, collate_fn=collate_fn_test)
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    model = ResNet18(n_cls).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    #
    print('running inference')
    result = infer(test_dl, model)
    #
    sub = {'id_code': test_df.id_code,
           'diagnosis': result}
    sub = pd.DataFrame(sub)
    sub.to_csv('submission.csv', index=False)
    print(f'result saved to submission.csv')
