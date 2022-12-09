from __future__ import print_function, division
import os, glob
import torch
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import filters, transform


class ActinDataset(Dataset):
    """
    piss
    """

    def __init__(self, root_dir, transform=None):
        """
        piss
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len([
            name for name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, name))
                                                          and name.split(".")[-1] != "csv"
        ])

    def __getitem__(self, item):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # print(":)")
        # print(item)

        img_name = os.path.join(self.root_dir, str(item))
        image = np.load(img_name)["arr_0"]
        sample = {'image': image}

        if self.transform:
            # sample = self.transform(sample)
            sample = {'image': self.transform(sample['image'])}

        return sample

class CompleteImageDataset(Dataset):
    """
    Creates a `Dataset` that allows to extract crops from complete images.
    """
    def __init__(self, root_dir, size=224, step=0.75, chan=0, transform=None,
                    preload_cache=True):
        """
        Instantiates a `CompleteImageDataset`
        """
        self.root_dir = root_dir
        self.size = size
        self.step = step
        self.preload_cache = preload_cache
        self.chan = chan
        self.transform = transform

        self.data_info = []
        self.cache = {}

        self.get_data_info()

    def get_data_info(self):
        """
        Gets all images from the root directory
        """
        image_names = glob.glob(os.path.join(self.root_dir, "**/*.tif"), recursive=True)

        for image_name in tqdm.tqdm(image_names, desc="Images"):
            image = tifffile.imread(image_name)

            # Rescales from 15nm pixels to 20nm pixels
            image = transform.rescale(image, 15 / 20, preserve_range=True).astype(np.uint16)

            foreground = np.sum(image, axis=0)
            foreground = filters.gaussian(foreground, sigma=5)
            threshold = filters.threshold_otsu(foreground)
            foreground = foreground > threshold

            for j in range(0, image.shape[-2] - self.size, int(self.step * self.size)):
                for i in range(0, image.shape[-1] - self.size, int(self.step * self.size)):

                    foreground_crop = foreground[j : j + self.size, i : i + self.size]

                    if foreground_crop.sum() > 0.1 * foreground_crop.size:
                        crop = image[self.chan, j : j + self.size, i : i + self.size]

                        self.data_info.append({
                            "key" : image_name,
                            "chan" : self.chan,
                            "pos" : (j, i),
                            "slc" : tuple((
                                self.chan,
                                slice(j, j + self.size),
                                slice(i, i + self.size)
                            ))
                        })

            if self.preload_cache:
                self.cache[image_name] = image

    def get_data(self, index):
        info = self.data_info[index]
        if info["key"] in self.cache:
            image = self.cache[info["key"]]
            slc = info["slc"]
        else:
            image_name = info["key"]
            slc = info["slc"]
            image = tifffile.imread(image_name)

            # Rescales from 15nm pixels to 20nm pixels
            image = transform.rescale(image, 15 / 20, preserve_range=True).astype(np.uint16)

            # Clears cache and updates with most recent image
            self.cache = {}
            self.cache[image_name] = image
        return image[slc]

    def __getitem__(self, index):
        image = self.get_data(index)

        image = (image - image.min()) / (image.max() - image.min())
        image = image.astype(np.float32)

        sample = {'image': image}
        if self.transform:
            sample = {'image': self.transform(sample['image'])}

        return sample

    def __len__(self):
        return len(self.data_info)

if __name__ == "__main__":

    # QUALITY_THRESHOLD = 0.7
    # train_path = f"./data/actin/train_quality_{QUALITY_THRESHOLD}"
    # test_path = f"./data/actin/test_quality_{QUALITY_THRESHOLD}"
    #
    # all_trains = [name for name in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, name))]
    # all_tests = [name for name in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, name))]
    #
    # train_set = ActinDataset(train_path)
    # test_set = ActinDataset(test_path)
    #
    # print(len(all_trains), len(train_set))
    # print(len(all_tests), len(test_set))
    # # print(img.shape)


    dataset = CompleteImageDataset("./results/data/PSD95-Bassoon", preload_cache=True)
    print(len(dataset))
    print(dataset[0])
