"""
- Go through the train and test folders of a given dataset, extract the images with q >= th and place them in a folder
- save a .csv containing the idx name of the file and corresponding quality
- Run model on all imgs in folder
- Save the output datamap and processed datamap to folders
- Repeat for every dataset
"""
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
from dataset import CompleteImageDataset
from unet import UNet
from skimage import filters

DENOISING_THRESHOLD = 0.25

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", action="store_true",
                        help="enables cuda GPU")
    parser.add_argument("--folder", required=True, type=str,
                        help="enables cuda GPU")
    args = parser.parse_args()

    if args.folder.endswith("/"):
        args.folder = args.folder[:-1]

    # load model
    model_path = "./results/models/20220726-095941_udaxihhe/best.pt"

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

    # build dataloader
    dataset = CompleteImageDataset(args.folder)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=32, drop_last=False)

    datamaps_path = os.path.join("./results/datamaps", os.path.basename(args.folder))
    os.makedirs(datamaps_path, exist_ok=True)

    datamaps_processed_path = os.path.join("./results/datamaps_processed", os.path.basename(args.folder))
    os.makedirs(datamaps_processed_path, exist_ok=True)

    print(f"Running model through images at {args.folder}")
    # run model through dataloader
    for i, X in enumerate(tqdm(dataloader, desc="[----] ")):
        # Reshape and send to gpu
        X = X["image"]

        if X.dim() == 3:
            X = X.unsqueeze(1)
        if args.cuda:
            X = X.cuda()

        # Prediction and loss computation
        datamap = unet.forward(X).cpu().data.numpy()

        for n, dmap in enumerate(datamap):
            dmap = dmap.squeeze()
            num = i * dataloader.batch_size + n
            np.save(os.path.join(datamaps_path, f"{num:06d}"), dmap)
            datamap_processed = np.where(datamap < DENOISING_THRESHOLD, 0, dmap)
            np.save(os.path.join(datamaps_processed_path, f"{num:06d}"), datamap_processed)

        # To avoir memory leak
        del datamap
        del datamap_processed
        del X
