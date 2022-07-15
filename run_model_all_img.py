"""
the goal here is to run the model on all the images (train_all and test_all) and look at the resulting datamaps
- make sure I keep track of the quality rating of all images so I can plot MSE between real acq - pySTED acq on datamap
  as a function of quality rating
- anything else?
"""

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


def data_splitter_verif(output_path, dry_run=False):
    if dry_run:
        output_path = "./data/actin/verif_dry"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        shutil.rmtree(output_path)
        os.mkdir(output_path)

    train_all_path = "./data/actin/train_all"
    test_all_path = "./data/actin/test_all"

    curr_idx = 0
    idx_list, quality_list = [], []
    for idx, name in enumerate(os.listdir(train_all_path)):
        if dry_run and curr_idx > 5:
            break
        if name.split(".")[-1] == "npz":
            quality = float(name.split("-")[-1][:-4])

            img = np.load(train_all_path + "/" + name)["arr_0"]
            unnormalized_img = np.floor(img * (2 ** 16 - 1) - 2 ** 15)
            normalized_img = unnormalized_img / np.max(unnormalized_img)
            np.savez(output_path + f"/{curr_idx}.npz", normalized_img)
            os.rename(output_path + f"/{curr_idx}.npz", output_path + f"/{str(curr_idx)}")

            idx_list.append(curr_idx)
            quality_list.append(quality)
            curr_idx += 1

    for idx, name in enumerate(os.listdir(test_all_path)):
        if dry_run and curr_idx > 5:
            break
        if name.split(".")[-1] == "npz":
            quality = float(name.split("-")[-1][:-4])

            img = np.load(test_all_path + "/" + name)["arr_0"]
            unnormalized_img = np.floor(img * (2 ** 16 - 1) - 2 ** 15)
            normalized_img = unnormalized_img / np.max(unnormalized_img)
            np.savez(output_path + f"/{curr_idx}.npz", normalized_img)
            os.rename(output_path + f"/{curr_idx}.npz", output_path + f"/{str(curr_idx)}")

            idx_list.append(curr_idx)
            quality_list.append(quality)
            curr_idx += 1

    # iterate through all files in test_all / train_all
    # copy img over to new folder with name == i
    # build dataframe with a column for img idx, a column for img quality
    data = {"idx": idx_list, "quality": quality_list}
    dataframe = pd.DataFrame(data=data)
    dataframe.to_csv(output_path + "/img_quality.csv")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", action="store_true",
                        help="Performs a dry run")
    parser.add_argument("-c", "--cuda", action="store_true",
                        help="enables cuda GPU")
    args = parser.parse_args()

    data_splitter_verif("./data/actin/verif_all", dry_run=args.dry_run)

    model_path = "./data/actin/Results/generation/20220712-115736_udaxihhe/models/best.pt"

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

    if args.dry_run:
        data_path = "./data/actin/verif_dry"
    else:
        data_path = "./data/actin/verif_all"

    datamap_output_path = "./data/actin/verif_datamaps"
    if not os.path.exists(datamap_output_path):
        os.mkdir(datamap_output_path)

    all_data = ActinDataset(data_path)

    all_data_loader = DataLoader(all_data)

    random_batch = random.randint(0, len(all_data_loader) - 1)
    for i, X in enumerate(tqdm(all_data_loader, desc="[----] ")):
        # Reshape and send to gpu
        X = X["image"]
        if X.dim() == 3:
            X = X.unsqueeze(0)
        if args.cuda:
            X = X.cuda()

        # Prediction and loss computation
        unet_dmap_pred = unet.forward(X).cpu().data.numpy()[0, 0]

        np.save(datamap_output_path + f"/{i}", unet_dmap_pred)

        # To avoir memory leak
        del unet_dmap_pred
        del X
