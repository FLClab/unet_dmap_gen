"""
- Go through all tif images in the specified folder
- Run model on all available crops in each images
- Save the output and the filtered output
"""
import os
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tifffile

from torch.utils.data import DataLoader, Dataset
from torch import nn
from tqdm import *
from data.split_data_quality import data_splitter, data_splitter_v2
from collections import defaultdict
from dataset import get_image_dataset
from unet import UNet
from skimage import filters

DENOISING_THRESHOLD = 0.25

def savefile(savename, image):
    """
    Saves an image to disk.

    :param savename: A `str` of the filename
    :param image: A `numpy.ndarray` of the image
    """
    # Creates folder architecture if not available
    os.makedirs(os.path.dirname(savename), exist_ok=True)

    if savename.endswith(".tif") or savename.endswith(".tiff"):
        tifffile.imwrite(savename, image.astype(np.float32))
    else:
        np.save(savename, image)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", action="store_true",
                        help="enables cuda GPU")
    parser.add_argument("--model_path", type=str, default="./results/models/20220726-095941_udaxihhe/best.pt",
                        help="Path of the model to use for prediction")                        
    parser.add_argument("--folder", required=True, type=str,
                        help="Name of folder to analyse")
    parser.add_argument("--keep-name", action="store_true",
                        help="Whether the name of the files should be kept the same")
    args = parser.parse_args()

    if args.folder.endswith("/"):
        args.folder = args.folder[:-1]

    # load model
    model_path = args.model_path

    unet = UNet(n_channels=1, n_classes=1)
    if not args.cuda:
        print("no cuda")
        unet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print("cuda")
        unet.load_state_dict(torch.load(model_path))
        device = torch.device("cuda")
        unet.to(device)
    # use the unet in eval mode
    unet.eval()

    # build dataloader
    dataset = get_image_dataset(args.folder, chan=0, preload_cache=True)
    dataloader = DataLoader(dataset, num_workers=1, batch_size=16, drop_last=False)

    datamaps_path = os.path.join("./results/datamaps", os.path.basename(args.folder))
    os.makedirs(datamaps_path, exist_ok=True)

    datamaps_processed_path = os.path.join("./results/datamaps_processed", os.path.basename(args.folder))
    os.makedirs(datamaps_processed_path, exist_ok=True)

    print(f"Running model through images at {args.folder}")
    # run model through dataloader
    for i, X in enumerate(tqdm(dataloader, desc="[----] ")):
        # Reshape and send to gpu
        info = X["info"]
        X = X["image"]

        if X.dim() == 3:
            X = X.unsqueeze(1)
        if args.cuda:
            X = X.cuda()

        # Prediction and loss computation
        datamap = unet.forward(X).cpu().data.numpy()

        if args.keep_name:
            for key, dmap in zip(info["key"], datamap):
                dmap = dmap.squeeze()

                savepath = key.replace(args.folder, datamaps_path)
                savefile(savepath, dmap)

                savepath = key.replace(args.folder, datamaps_processed_path)
                datamap_processed = np.where(dmap < DENOISING_THRESHOLD, 0, dmap)
                savefile(savepath, datamap_processed)
        else:
            for n, dmap in enumerate(datamap):
                dmap = dmap.squeeze()
                num = i * dataloader.batch_size + n
                savefile(os.path.join(datamaps_path, f"{num:06d}"), dmap)
                datamap_processed = np.where(dmap < DENOISING_THRESHOLD, 0, dmap)
                savefile(os.path.join(datamaps_processed_path, f"{num:06d}"), datamap_processed)

        # To avoir memory leak
        del datamap
        del datamap_processed
        del X
