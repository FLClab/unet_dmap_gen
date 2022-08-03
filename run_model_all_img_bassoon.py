import os
import numpy as np
import pandas as pd
import torch
import random
import datetime
import shutil
import json
# import loader
import utils
import time
import pickle
import string
import math

from torch.utils.data import DataLoader, Dataset
from torch import nn
from tqdm import *
from data.split_data_quality import data_splitter, data_splitter_v2
from collections import defaultdict
from dataset import ActinDataset
from unet import UNet
from skimage import filters


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dry-run", action="store_true",
                    help="Performs a dry run")
parser.add_argument("-c", "--cuda", action="store_true",
                    help="enables cuda GPU")
args = parser.parse_args()

data_path = "./data/factin_patches"
model_path = "./data/big_dataset/Results/generation/20220726-095941_udaxihhe/models/best.pt"
datamap_output_path = "./data/factin_patches_datamaps"
if not os.path.exists(datamap_output_path):
    os.mkdir(datamap_output_path)

unet = UNet(n_channels=1, n_classes=1)
# unet.load_state_dict(torch.load(model_path))
if not args.cuda:
    # unet = torch.load(model_path, map_location=torch.device('cpu'))
    unet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
else:
    # unet = torch.load(model_path)
    unet.load_state_dict(torch.load(model_path))
# use the unet in eval mode
unet.eval()

dataset = ActinDataset(data_path)
data_loader = DataLoader(dataset)

random_batch = random.randint(0, len(data_loader) - 1)
for i, X in enumerate(tqdm(data_loader, desc="[----] ")):
    # Reshape and send to gpu
    X = X["image"]
    if X.dim() == 3:
        X = X.unsqueeze(0).float()
    if args.cuda:
        X = X.cuda().float()

    # Prediction and loss computation
    unet_dmap_pred = unet.forward(X).cpu().data.numpy()[0, 0]

    np.save(datamap_output_path + f"/{i}", unet_dmap_pred)

    # To avoir memory leak
    del unet_dmap_pred
    del X
