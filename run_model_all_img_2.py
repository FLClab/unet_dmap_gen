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


def data_splitter_all_dataset(dataset_paths_list, output_path="./data/big_dataset/run_all"):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    data_save_path = output_path + "/all_data"
    if not os.path.exists(data_save_path):
        os.mkdir(data_save_path)

    data = {
        "idx": [],
        "quality": [],
        "dataset": []
    }

    datasets_list = ["actin", "camkii", "lifeact", "psd", "tubulin"]
    counter = 0
    for path in dataset_paths_list:
        dataset = path.split("/")[2]
        if dataset not in datasets_list:
            raise ValueError("uh oh")
        img_files = [
            f for f in os.listdir(path) if
            os.path.isfile(os.path.join(path, f)) and f.split(".")[-1] == "npz"
        ]
        # iterate through the list
        for img in img_files:
            quality = int(img.split(".")[1]) / 1000
            unnormalized_img = np.load(os.path.join(path, img))["arr_0"]
            normalized_img = unnormalized_img / np.max(unnormalized_img)   # not the good normalization

            np.savez(data_save_path + f"/{counter}.npz", normalized_img)
            os.rename(data_save_path + f"/{counter}.npz", data_save_path + f"/{str(counter)}")
            data["idx"].append(counter)
            data["quality"].append(quality)
            data["dataset"].append(dataset)
            counter += 1

    dataframe = pd.DataFrame(data)
    dataframe.to_csv(data_save_path + "/info.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", action="store_true",
                        help="enables cuda GPU")
    args = parser.parse_args()

    all_subset_paths = [
        "./data/actin/train_all", "./data/camkii/CaMKII_Neuron/train", "./data/lifeact/LifeAct_Neuron/train",
        "./data/psd/PSD95_Neuron/train", "./data/tubulin/train", "./data/actin/test_all",
        "./data/camkii/CaMKII_Neuron/test", "./data/lifeact/LifeAct_Neuron/test",
        "./data/psd/PSD95_Neuron/test", "./data/tubulin/test"
    ]

    # all_subset_paths = ["./data/actin/test_ti"]

    all_data_path = "./data/big_dataset/run_all"
    if os.path.exists(all_data_path):
        shutil.rmtree(all_data_path)
    data_splitter_all_dataset(all_subset_paths, output_path=all_data_path)

    model_path = "./data/big_dataset/Results/generation/20220725-120330_udaxihhe/models/best.pt"

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

    all_data = ActinDataset(all_data_path + "/all_data")
    all_data_loader = DataLoader(all_data)

    output_save_path = all_data_path + "/all_output"
    if not os.path.exists(output_save_path):
        os.mkdir(output_save_path)

    for i, X in enumerate(tqdm(all_data_loader, desc="[----] ")):
        # Reshape and send to gpu
        X = X["image"]
        if X.dim() == 3:
            X = X.unsqueeze(0)
        if args.cuda:
            X = X.cuda()

        # Prediction and loss computation
        unet_dmap_pred = unet.forward(X).cpu().data.numpy()[0, 0]

        np.save(output_save_path + f"/{i}", unet_dmap_pred)

        # To avoir memory leak
        del unet_dmap_pred
        del X

