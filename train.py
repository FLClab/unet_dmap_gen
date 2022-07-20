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
from data.split_data_quality import data_splitter, data_splitter_v2, big_dataset_builder
from collections import defaultdict
from dataset import ActinDataset
from unet import UNet
from skimage import filters


def gaussian_fn(M, std):
    # taken from last comment on https://stackoverflow.com/questions/60534909/gaussian-filter-in-pytorch
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w


def create_savefolder(params, dry_run=False):
    """
    Creates the savefolder and returns the path
    :param path: A `str` of path where to create the folder
    :param params: A `dict` of parameters
    :returns : A `str` of the created folder
    """
    random_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(8))
    output_folder = f"{params['datetime']}_{random_string}"
    # output_folder = f"{params['datetime']}"
    if dry_run:
        output_folder = "dryrun"
    output_folder = os.path.join(params["savefolder"], output_folder)
    output_folder_models = os.path.join(output_folder, "models")
    try :
        if dry_run:
            if os.path.isdir(output_folder):
                shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(output_folder_models, exist_ok=True)
    except OSError as err:
        print("The name of the folder already exist! Try changing the name of the folder.")
        print("OSError", err)
        exit()
    return output_folder


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 2d tensor. Filtering is performed
    seperately for each channel in the input using a depthwise convolution.
    :param channels: Number of channels of the input data
    :param sigma: Sigma parameter of the gaussian filter
    :param dim: The number of dimension of the data (2D). Defaults to 2D
    """
    def __init__(self, channels, sigma, dim=2):
        """
        Instantiates the `GaussianSmoothing`
        """
        super(GaussianSmoothing, self).__init__()

        self.dim = dim

        sigma = [sigma] * dim
        # Kernel size should encompass 3 sigma and should be odd
        kernel_size = [np.ceil(sig * 2 * 3) // 2 * 2 + 1 for sig in sigma]
        self.kernel_size = kernel_size[0]

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 2:
            self.conv = nn.functional.conv2d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def amax(self, x, dim, keepdim=True):
        """
        Implements an `amax` method that is not present in PyTorch 1.5
        :param x: An input `torch.tensor`
        :param dim: An int or tuple of ints to apply the max operator
        :param keepdim: Wheter the output `torch.tensor` should have the same dims
        :returns : A `torch.tensor`
        """
        if isinstance(dim, int):
            return torch.max(x, dim=dim, keepdim=keepdim)[0]
        elif isinstance(dim, (tuple, list)):
            for d in dim:
                x = torch.max(x, dim=d, keepdim=keepdim)[0]
            return x

    def forward(self, x, normalize=None):
        """
        Apply gaussian filter to input.
        :param x: A `torch.tensor` of the input data
        :param normalize: (optional) A `str` of the normalization type
        :returns : A `torch.tensor` of the filtered input data
        """
        pad = int((self.kernel_size - 1) // 2)
        x = nn.functional.pad(x, (pad, pad, pad, pad), mode='reflect')
        x = self.conv(x, weight=self.weight, groups=self.groups)
        if normalize == "max":
            x = x / (self.amax(x, dim=(-2, -1), keepdim=True) + 0.00001)
        elif normalize == "kernel":
            x = x / self.weight[0, 0, int((self.kernel_size - 1) / 2), int((self.kernel_size - 1) / 2)]
            x = torch.clamp(x, 0, 1)
        else:
            pass
        return x


class GaussianConv(nn.Module):
    def __init__(self):
        gaussian_kernel = gkern()
        super(GaussianConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                1, 1, gaussian_kernel.shape[0], padding=int(gaussian_kernel.shape[0] / 2), bias=False
            )
        )

    def forward(self, X):
        return self.conv(X)


def gkern(kernlen=256, std=128):
    # taken from last comment on https://stackoverflow.com/questions/60534909/gaussian-filter-in-pytorch
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    gkern2d = gkern2d.unsqueeze(0)
    return gkern2d


def criterion(input, target, sigma=1, cuda=False):
    """
    loss is mse (?) between input img and traget convolved with gaussian filter
    """

    gkern = GaussianSmoothing(1, sigma)
    if cuda:
        gkern = gkern.cuda()
    if cuda:
        target = target.to(device='cuda')
    psf_target = gkern.forward(target, normalize=None)   # TODO : normalement c'était none, tester avec kernel
    if cuda:
        psf_target = psf_target.to(device='cuda')

    # Quadratic loss
    mse_loss_array = (input - psf_target)**2
    mse_loss = mse_loss_array.sum() / (mse_loss_array.shape[-1] * mse_loss_array.shape[-2])

    return mse_loss


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", action="store_true",
                        help="Performs a dry run")
    parser.add_argument("-c", "--cuda", action="store_true",
                        help="enables cuda GPU")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sets the default random seed")
    parser.add_argument("--no-tqdm", action="store_true",
                        help="Wheter to use tqdm")
    parser.add_argument("--quality-threshold", type=float, default=0.7)
    parser.add_argument("--sigma", type=float, default=1.6985)
    args = parser.parse_args()

    # Setting the seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    PATH = "./data/big_dataset"

    if args.no_tqdm:
        def tqdm(loader, *args, **kwargs):
            return loader

    if args.dry_run:
        training_path = PATH + f"/train_dry"
        testing_path = PATH + f"/test_dry"
    else:
        # data_splitter_v2(PATH, quality_threshold=args.quality_threshold, individual_norm=True, rotations=True)
        train_subset_paths = [
            "./data/actin/train_all", "./data/camkii/CaMKII_Neuron/train", "./data/lifeact/LifeAct_Neuron/train",
            "./data/psd/PSD95_Neuron/train", "./data/tubulin/train"
        ]
        test_subset_paths = [
            "./data/actin/test_all", "./data/camkii/CaMKII_Neuron/test", "./data/lifeact/LifeAct_Neuron/test",
            "./data/psd/PSD95_Neuron/test", "./data/tubulin/test"
        ]
        big_dataset_builder(
            train_subset_paths, test_subset_paths, quality_threshold=args.quality_threshold, output_path=PATH
        )
        training_path = PATH + f"/train_quality_{args.quality_threshold}"
        testing_path = PATH + f"/test_quality_{args.quality_threshold}"

    training, testing = ActinDataset(training_path), ActinDataset(testing_path)

    # metadata = utils.load_json("./dataset/metadata_2019-06-26.json")   # j'ai pas ça
    # maxima, minima = [], []
    # for key, value in metadata.items():
    #     maxima.append(value["max"])
    #     minima.append(value["min"])
    # image_maximum = np.median(maxima, axis=0)
    # image_minimum = np.median(minima, axis=0)
    image_maximum = None,
    image_minimum = None

    lr, epochs = 1e-4, 1000   # 1000
    min_valid_loss = np.inf

    stats = defaultdict(list)

    # do I actually need all this?
    trainer_params = {
        "savefolder": f"{PATH}/Results/generation",
        "datetime": datetime.datetime.today().strftime("%Y%m%d-%H%M%S"),
        "dry_run": args.dry_run,
        "size": 128,
        "pretrained_model": None,
        "pretrained_model_ckpt": "best",
        "image_min": image_minimum,   #.tolist(),
        "image_max": image_maximum,   #.tolist(),
        "len_training": len(training),
        "len_validation": None,
        "len_testing": len(testing),
        "training_path": training_path,
        "validation_path": None,
        "testing_path": testing_path,
        "lr": lr,
        "epochs": epochs,
        "batch_size": 32,
        "cuda": args.cuda,
        "data_aug": 0.5,
        "step": 96,
        "seed": args.seed,
        "classes": None,
        "omit_transforms": ["periodical_transform", "swap_channels", "shear", "elastic_transform", "intensity_scale",
                            "gamma_adaptation"],
        "protein_encoder_model_params": {
            "use": True,
            "in_channels": [0],
            "number_filter": 4,
            "depth": 6,
            "latent_dim": 256
        },
        "structure_encoder_model_params": {
            "use": False,
            "in_channels": [0, 1],
            "number_filter": 4,
            "depth": 6,
            "latent_dim": None
        },
        "decoder_model_params": {
            "use": True,
            "out_channels": [0],
            "number_filter": 4,
            "depth": 6,
            "latent_dim": 256,
            "single_bottleneck": True
        },
        "estopper": {
            "patience": 10,
            "threshold": 1.
        },
        "scheduler": {
            "patience": 1000,
            "threshold": 0.005,
            "min_lr": 1e-5,
            "factor": 0.5,
            "verbose": True
        },
        "criterion": {
            "name": "MSELoss",
            "kwargs": {
            }
        },
        "logging": {
            "collage_luts": [["green", "red", "blue"], ["green", "red", "blue"], ["green"]],
            "collage_drange": None
        }
    }

    output_folder = create_savefolder(trainer_params, dry_run=trainer_params["dry_run"])
    json.dump(trainer_params, open(os.path.join(output_folder, "trainer_params.json"), "w"), indent=4, sort_keys=True)

    train_loader = DataLoader(training)
    valid_loader = DataLoader(testing)

    unet = UNet(n_channels=1, n_classes=1)   # right? og code uses trainer_params as input

    if isinstance(trainer_params["pretrained_model"], str):
        unet.load_state_dict(
            torch.load(
                os.path.join(trainer_params["savefolder"], trainer_params["pretrained_model"])
            )
        )

    if trainer_params["cuda"]:
        unet = unet.cuda()

    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
    if isinstance(trainer_params["pretrained_model"], str):
        optimizer.load_state_dict(optimizers["optimizer"])

    criterions = {   # not sure about this
        "GaussianConvolutionLoss": criterion
    }

    estopper = utils.EarlyStopper(**trainer_params["estopper"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **trainer_params["scheduler"])

    os.path.join(output_folder, "training_parameters")
    with open(f'{os.path.join(output_folder, "training_parameters")}.pkl', 'wb') as handle:
        pickle.dump(trainer_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # TODO : do this in the data splitter func
    # rename files 0 to n-1 in order to load them correctly
    files_training = [name for name in os.listdir(training_path) if os.path.isfile(os.path.join(training_path, name))]
    files_testing = [name for name in os.listdir(testing_path) if os.path.isfile(os.path.join(testing_path, name))]
    for i, name in enumerate(sorted(files_training)):
        os.rename(os.path.join(training_path, name), os.path.join(training_path, f"{str(i)}"))
    for i, name in enumerate(sorted(files_testing)):
        os.rename(os.path.join(testing_path, name), os.path.join(testing_path, f"{str(i)}"))

    dataframe = pd.DataFrame({"epoch": [None], "trainMean": [None], "testMean": [None], "testMin": [None]})
    dataframe.drop([0])
    log_file_path = f"{os.path.join(output_folder, 'training_log')}.txt"
    log_file = open(log_file_path, "a")

    # TRAINING LOOP
    for epoch in range((3 if args.dry_run else epochs)):
        start = time.time()
        print("[----] Starting epoch {}/{}".format(epoch, epochs))
        log_file.write("[----] Starting epoch {}/{} \n".format(epoch, epochs))

        # Keep track of the loss of train and test
        statLossTrain, statLossTest = [], []

        # Puts the network in training mode
        unet.train()

        random_batch = random.randint(0, len(train_loader))
        for i, X in enumerate(tqdm(train_loader, desc="[----] ")):

            # Reshape and send to gpu
            X = X["image"]
            if X.dim() == 3:
                X = X.unsqueeze(0)
            if trainer_params["cuda"]:
                X = X.cuda()

            # New batch we reset the optimizer
            optimizer.zero_grad()

            # Prediction and loss computation
            unet_dmap_pred = unet.forward(X)
            loss = criterion(X, unet_dmap_pred, sigma=args.sigma, cuda=args.cuda)

            # Keeping track of statistics
            statLossTrain.append([loss.item()])

            # Back-propagation and optimizer step
            loss.backward()

            optimizer.step()

            if (i == random_batch) & ((epoch % 10 == 0) or (epoch < 10)):
                Xnumpy = X.cpu().data.numpy()
                unet_dmap_pred_numpy = unet_dmap_pred.cpu().data.numpy()

                if not os.path.exists(os.path.join(output_folder, "training_examples")):
                    os.mkdir(os.path.join(output_folder, "training_examples"))
                np.save(os.path.join(output_folder, "training_examples", f"epoch_{epoch}_input_img"), Xnumpy)
                np.save(os.path.join(output_folder, "training_examples", f"epoch_{epoch}_output_dmap"),
                        unet_dmap_pred_numpy)

                del Xnumpy
                del unet_dmap_pred_numpy

                # To avoid memory leak
            del X,  loss

        # Puts the network in training mode
        unet.eval()

        random_batch = random.randint(0, len(valid_loader))
        for i, X in enumerate(tqdm(valid_loader, desc="[----] ")):

            # Reshape and send to gpu
            X = X["image"]
            if X.dim() == 3:
                X = X.unsqueeze(0)
            if trainer_params["cuda"]:
                X = X.cuda()

            # Prediction and loss computation
            unet_dmap_pred= unet.forward(X)

            # Calculates the error in generation
            loss = criterion(X, unet_dmap_pred, sigma=args.sigma, cuda=args.cuda)

            # Keeping track of statistics
            statLossTest.append([loss.item()])

            if (i == random_batch) & ((epoch % 10 == 0) or (epoch < 10)):
                Xnumpy = X.cpu().data.numpy()
                unet_dmap_pred_numpy = unet_dmap_pred.cpu().data.numpy()

                if not os.path.exists(os.path.join(output_folder, "validation_examples")):
                    os.mkdir(os.path.join(output_folder, "validation_examples"))
                np.save(os.path.join(output_folder, "validation_examples", f"epoch_{epoch}_input_img"), Xnumpy)
                np.save(os.path.join(output_folder, "validation_examples", f"epoch_{epoch}_output_dmap"),
                        unet_dmap_pred_numpy)

                del Xnumpy
                del unet_dmap_pred_numpy

            # To avoid memory leak
            del X, loss

        # Aggregate stats
        for key, func in zip(("trainMean", "trainMed", "trainMin", "trainStd"),
                             (np.mean, np.median, np.min, np.std)):
            stats[key].append(func(statLossTrain, axis=0))
        for key, func in zip(("testMean", "testMed", "testMin", "testStd"),
                             (np.mean, np.median, np.min, np.std)):
            stats[key].append(func(statLossTest, axis=0))
        stats["lr"].append([
            optimizer.param_groups[0]["lr"]
        ])
        log_file.write(f"Loss/Train at epoch {epoch} : {stats['trainMean'][-1][0]} \n")
        log_file.write(f"Loss/Validation at epoch {epoch} : {stats['testMean'][-1][0]} \n")
        log_file.write(f"Loss/min_Validation at epoch {epoch} : {np.min(stats['testMean'][0], axis=0)} \n")
        latest_data = {
            "epoch": epoch,
            "trainMean": stats['trainMean'][-1][0],
            "testMean": stats['testMean'][-1][0],
            "testMin": np.min(stats['testMean'][0], axis=0)
        }
        dataframe = dataframe.append(latest_data, ignore_index=True)
        dataframe.to_csv(f"{output_folder}/training_progress.csv")

        log_file.write(f"LearningRate/network at epoch {epoch} : {stats['lr'][-1][0]} \n")

        scheduler.step(np.min(stats["testMean"]))
        estopper.step(np.mean(statLossTrain), np.mean(statLossTest))

        # Save if best network so far
        if stats["testMean"][-1] < min_valid_loss:
            torch.save(unet.state_dict(), f"{os.path.join(output_folder, 'models')}/best.pt")

        # Save every ten epoch the state of the network
        if epoch % 10 == 0:
            networks = {
                "unet": unet
            }
            optimizers = {
                "optimizer": optimizer
            }
            torch.save(unet.state_dict(), f"{os.path.join(output_folder, 'models')}/epoch_{epoch}.pt")

        print("[----] Epoch {} done!".format(epoch + 1))
        print("[----]     Avg loss train/validation : {} / {}".format(stats["trainMean"][-1], stats["testMean"][-1]))
        print("[----]     Current best network : {}".format(stats["testMin"][-1]))
        print("[----]     Current learning rate : {}".format(stats["lr"][-1]))
        print("[----]     Took {} seconds".format(time.time() - start))
        print("[----]     Current time : {}".format(datetime.datetime.now()))
        log_file.write("[----] Epoch {} done! \n".format(epoch + 1))
        log_file.write("[----]     Avg loss train/validation : {} / {} \n".format(stats["trainMean"][-1], stats["testMean"][-1]))
        log_file.write("[----]     Current best network : {} \n".format(stats["testMin"][-1]))
        log_file.write("[----]     Current learning rate : {} \n".format(stats["lr"][-1]))
        log_file.write("[----]     Took {} seconds \n".format(time.time() - start))
        log_file.write("[----]     Current time : {}".format(datetime.datetime.now()))
        pickle.dump(stats, open(os.path.join(output_folder, "stats.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    log_file.close()
