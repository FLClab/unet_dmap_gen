from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


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


if __name__ == "__main__":

    QUALITY_THRESHOLD = 0.7
    train_path = f"./data/actin/train_quality_{QUALITY_THRESHOLD}"
    test_path = f"./data/actin/test_quality_{QUALITY_THRESHOLD}"

    all_trains = [name for name in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, name))]
    all_tests = [name for name in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, name))]

    train_set = ActinDataset(train_path)
    test_set = ActinDataset(test_path)

    print(len(all_trains), len(train_set))
    print(len(all_tests), len(test_set))
    # print(img.shape)
