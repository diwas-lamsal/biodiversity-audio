import argparse
import sys
from copy import copy
import importlib
import torch
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from timm.scheduler import CosineLRScheduler

from transformations import get_transformations
from train_utils import init_logger, get_device, set_seed, train_loop
from sampler import MultilabelBalancedRandomSampler
from dataset import BirdClef2023Dataset
from model import AttModel

sys.path.append('./configs')

parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename", default="ait_bird_local")
parser_args, _ = parser.parse_known_args(sys.argv)
CFG = copy(importlib.import_module(parser_args.config).cfg)

TRAIN_AUDIO_PATH = CFG.train_audio_dir
TRAIN_DF_PATH = CFG.train_df_dir
NUM_CLASSES = CFG.num_classes


torch.set_flush_denormal(True)


logger = init_logger(log_file=f"logs/train_{CFG.exp_name}.log")
device = get_device()
set_seed(CFG.seed)
os.makedirs(os.path.join(CFG.model_output_path), exist_ok=True)

with open(TRAIN_DF_PATH + CFG.train_pkl_file, 'rb') as f:
    train = pickle.load(f)

train["rating"] = np.clip(train["rating"] / train["rating"].max(), 0.1, 1.0)

logger.info(train.shape)

# main loop
fold = 0

logger.info("=" * 90)
logger.info(f"Fold {fold} Training")
logger.info("=" * 90)

# trn_df = train[train['fold'] != fold].reset_index(drop=True)
# val_df = train[train['fold'] == fold].reset_index(drop=True)
trn_df, val_df = train_test_split(train, test_size=0.2, random_state=42, stratify=train['primary_label'])

# Resetting the indices for both training and validation DataFrames
trn_df = trn_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

sampler = None
if CFG.use_sampler:
    one_hot_target = np.zeros(
        (trn_df.shape[0], len(CFG.target_columns)), dtype=np.float32
        )

    for i, label in enumerate(trn_df.primary_label):
        primary_label = CFG.bird2id[label]
        one_hot_target[i, primary_label] = 1.0

    sampler = MultilabelBalancedRandomSampler(
        one_hot_target,
        trn_df.index,
        class_choice="least_sampled"
        )

logger.info(trn_df.shape)
logger.info(trn_df['primary_label'].value_counts())
logger.info(val_df.shape)
logger.info(val_df['primary_label'].value_counts())

tr_transforms, taa_transforms = get_transformations(CFG)

loaders = {}
trn_dataset = BirdClef2023Dataset(
        data_path=CFG.train_audio_dir,
        period=CFG.period,
        secondary_coef=CFG.secondary_coef,
        train=True,
        df=trn_df,
        tr_transforms = tr_transforms,
        cfg = CFG,
)
loaders['train'] = torch.utils.data.DataLoader(
    trn_dataset,
    sampler=sampler,
    **CFG.loader_params['train']
)
val_dataset = BirdClef2023Dataset(
        data_path=CFG.train_audio_dir,
        period=5,
        secondary_coef=CFG.secondary_coef,
        train=False,
        df=val_df,
        cfg = CFG,
)
loaders['valid'] = torch.utils.data.DataLoader(
    val_dataset,
    **CFG.loader_params['valid']
)

model = AttModel(
    backbone=CFG.backbone,
    num_class=CFG.num_classes,
    train_period=CFG.period,
    infer_period=5,
    cfg=CFG,
)

if CFG.pretrained_weights:
    model_path = CFG.pretrained_path
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    del checkpoint['state_dict']['head.weight']
    del checkpoint['state_dict']['head.bias']
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    params = list(model.parameters())
    print('the length of parameters is', len(params))
    for i in range(len(params)):
        params[i].data = torch.round(params[i].data*10**17) / 10**17
    del checkpoint

model = model.to(device)

criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CFG.lr_max,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=CFG.weight_decay,
    amsgrad=False,
    )
scheduler = CosineLRScheduler(
    optimizer,
    t_initial=10,
    warmup_t=1,
    cycle_limit=40,
    cycle_decay=1.0,
    lr_min=CFG.lr_min,
    t_in_epochs=True,
)

# # start training
train_loop(
    loaders['train'],
    loaders['valid'],
    model,
    optimizer,
    scheduler,
    criterion,
    epochs=CFG.epochs,
    fold=fold,
    device=device,
    taa_augmentation=taa_transforms,
    cfg=CFG,
    logger=logger,
   )

logger.info('training done!')