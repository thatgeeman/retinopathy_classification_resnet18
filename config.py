from collections import OrderedDict

# common params
vocab = OrderedDict(NoDR=0, Mild=1, Moderate=2, Severe=3, ProliferativeDR=4)
stats = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resize_sz = 384
bs = 64
show=False
# for train.py
drop_last = True
num_workers = 4  # number of workers for dataloader 
frac = 1.0  # fraction of samples to load from df
n_cls = len(vocab)
use_imagenet_wts = True
lr_head_f = 1e-2
lr_head_uf = 2e-4
lr_body_uf = 2e-6
# for infer.py

