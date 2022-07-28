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
from dataset import ActinDataset
from unet import UNet
from skimage import filters


def dataset_builder(dataset_paths, output_path, quality_th=0.7):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    info = {"idx": [], "quality": []}
    counter = 0
    for path in dataset_paths:
        print(f"Treating data at {path}")
        img_files = [
            f for f in os.listdir(path) if
            os.path.isfile(os.path.join(path, f)) and f.split(".")[-1] == "npz"
        ]
        # iterate through the list
        for img in img_files:
            quality = int(img.split(".")[1]) / 1000
            if quality >= quality_th:
                unnormalized_img = np.load(os.path.join(path, img))["arr_0"]
                normalized_img = (unnormalized_img - np.min(unnormalized_img)) / \
                                 (np.max(unnormalized_img) - np.min(unnormalized_img))

                np.savez(output_path + f"/{counter}.npz", normalized_img)
                os.rename(output_path + f"/{counter}.npz", output_path + f"/{str(counter)}")
                info["idx"].append(counter)
                info["quality"].append(quality)
                counter += 1

    dataframe = pd.DataFrame(info)
    dataframe.to_csv(output_path + "/info.csv")



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", action="store_true",
                        help="enables cuda GPU")
    args = parser.parse_args()

    quality_th = 0.7

    dataset_paths_list = [
        ["./data/actin_2/train", "./data/actin_2/test"],
        ["./data/camkii/CaMKII_Neuron/train", "./data/camkii/CaMKII_Neuron/test"],
        ["./data/lifeact/LifeAct_Neuron/train", "./data/lifeact/LifeAct_Neuron/test"],
        ["./data/psd/PSD95_Neuron/train", "./data/psd/PSD95_Neuron/test"],
        ["./data/tubulin/train", "./data/tubulin/test"]
    ]
    output_path_list = [
        f"./data/actin_2/all_quality_{quality_th}",
        f"./data/camkii/CaMKII_Neuron/all_quality_{quality_th}",
        f"./data/lifeact/LifeAct_Neuron/all_quality_{quality_th}",
        f"./data/psd/PSD95_Neuron/all_quality_{quality_th}",
        f"./data/tubulin/all_quality_{quality_th}",
    ]

    # load model
    model_path = "./data/big_dataset/Results/generation/20220726-095941_udaxihhe/models/best.pt"

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

    denoising_th = 0.25
    for dataset_paths, output_path in zip(dataset_paths_list, output_path_list):
        dataset_builder(dataset_paths, output_path, quality_th=quality_th)

        # build dataloader
        dataset = ActinDataset(output_path)
        dataloader = DataLoader(dataset)

        datamaps_path = output_path + "/datamaps"
        if not os.path.exists(datamaps_path):
            os.mkdir(datamaps_path)

        datamaps_processed_path = output_path + "/datamaps_processed"
        if not os.path.exists(datamaps_processed_path):
            os.mkdir(datamaps_processed_path)

        print(f"Running model through images at {output_path}")
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

            # To avoir memory leak
            del datamap
            del datamap_processed
            del X


