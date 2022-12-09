import os
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torch import nn
from tqdm import *
from data.split_data_quality import data_splitter, data_splitter_v2
from collections import defaultdict
from dataset import ActinDataset
from unet import UNet
from skimage import filters


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", action="store_true",
                        help="enables cuda GPU")
    args = parser.parse_args()

    # load model
    model_path = "./data/big_dataset/Results/generation/20220726-095941_udaxihhe/models/best.pt"

    unet = UNet(n_channels=1, n_classes=1)
    # unet.load_state_dict(torch.load(model_path))
    if not args.cuda:
        print("no cuda")
        # unet = torch.load(model_path, map_location=torch.device('cpu'))
        unet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print("cuda")
        # unet = torch.load(model_path)
        unet.load_state_dict(torch.load(model_path))
        device = torch.device("cuda")
        unet.to(device)
    # use the unet in eval mode
    unet.eval()

    denoising_th = 0.25

    data_path = os.path.expanduser("~/Documents/a22_docs/micranet_dataset_test")

    # build dataloader
    dataset = ActinDataset(data_path)
    dataloader = DataLoader(dataset)

    datamaps_path = data_path + "/datamaps"
    if not os.path.exists(datamaps_path):
        os.mkdir(datamaps_path)

    datamaps_processed_path = data_path + "/datamaps_processed"
    if not os.path.exists(datamaps_processed_path):
        os.mkdir(datamaps_processed_path)

    print(f"Running model through images at {data_path}")
    # run model through dataloader
    for i, X in enumerate(tqdm(dataloader, desc="[----] ")):
        # Reshape and send to gpu
        X = X["image"]
        if X.dim() == 3:
            X = X.unsqueeze(0)
        if args.cuda:
            X = X.cuda()

        # Prediction and loss computation
        datamap = unet.forward(X).cpu().data.numpy()[0, 0]

        np.save(datamaps_path + f"/{i}", datamap)
        datamap_processed = np.where(datamap < denoising_th, 0, datamap)
        np.save(datamaps_processed_path + f"/{i}", datamap_processed)

        # To avoir memory leak
        del datamap
        del datamap_processed
        del X
