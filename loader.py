import numpy
import os
import glob
import pandas
import random
import torch
import sys
import tqdm

from skimage import transform, filters
from scipy.ndimage import interpolation
from torch.utils.data import DataLoader, Sampler

import utils
from dlutils import augmenter
from dlutils.dataset import HDF5Dataset


class Dataset(HDF5Dataset):
    """
    Creates a dataset by inheriting the `HDF5Dataset`
    """

    def __init__(self, file_path, data_aug=0, validation=False, size=256, step=192,
                 classes=None, *args, **kwargs):
        """
        Inits the `Dataset`
        """
        super().__init__(file_path=file_path, load_data=kwargs["hdf5_params"]["load_data"],
                         data_cache_size=kwargs["hdf5_params"]["data_cache_size"])
        self.size = size
        self.step = step
        self.group_names = list(sorted([info["group_name"] for info in self.get_data_infos("label")]))
        self.data_augmentation = augmenter.DataAugmenter(data_aug if not validation else 0,
                                                         omit=kwargs["omit_transforms"])
        self.classes = classes
        self.valid_index = self.get_valid_index(classes)

    def get_valid_index(self, classes):
        """
        Gets the valid index of a specific classification
        :param classes: A `list` of valid classes
        :returns : A `numpy.ndarray` of valid index
        """
        if isinstance(classes, list):
            valid = []
            for index in tqdm.trange(self.cumsum_shapes[-1], desc="Valid index"):
                image, data_info = self.get_data("data", index)  # get data
                group_name = data_info["group_name"]

                if any([c in group_name.lower() for c in classes]):
                    valid.append(index)
            return numpy.array(valid)

        return numpy.arange(self.cumsum_shapes[-1])

    def get_data(self, type, i):
        """
        Call this function anytime you want to access a chunk of data from the
        dataset. This will make sure that the data is loaded in case it is
        not part of the data cache.
        """
        file_path_index = numpy.argmax(i < self.cumsum_shapes)
        data_info = self.get_data_infos(type)[file_path_index]

        key_cache = "-".join((data_info["file_path"], data_info["group_name"]))
        if key_cache not in self.data_cache:
            self._load_data(data_info["file_path"], data_info["group_name"])

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[file_path_index]['cache_idx']
        group_name = self.get_data_infos(type)[file_path_index]['group_name']

        # reshapes the dataset accordingly if dataset_attrs
        index_in_dataset = i % self.get_data_infos(type)[file_path_index]['shape'][0]
        output = self.data_cache[key_cache][cache_idx][index_in_dataset]
        if "shapes" in data_info["dataset_attrs"]:
            new_shape = data_info["dataset_attrs"]["shapes"][index_in_dataset]
            output = output[..., :new_shape[0], :new_shape[1]]
        data_info["index_in_dataset"] = index_in_dataset
        return output, data_info

    def __getitem__(self, index):
        """
        Reimplements the `__getitem__` from the `HDF5Dataset` class
        :param index: An `int` of the index to retreive from dataset
        """

        index = self.valid_index[index]

        image, data_info = self.get_data("data", index)  # get data
        group_name = data_info["group_name"]

        # classification = data_info["dataset_attrs"]["classes"][data_info["index_in_dataset"]]
        label, data_info = self.get_data("label", index)  # get label
        foreground = label[-1]

        image = image[[0, 1, 3]]  # Keeps DIO and homer signal
        sampled_slice = self.sample_slice(image)
        while foreground[sampled_slice[1:]].sum() <= 0.1 * self.size * self.size:
            sampled_slice = self.sample_slice(image)

        x = (image / 255.).astype(numpy.float32)
        x = x[sampled_slice]

        x, y = self.data_augmentation(x, foreground[sampled_slice[1:]])
        y = (y >= 0.5).astype(int)

        x = torch.tensor(x)
        y = torch.tensor(y)

        return x, y

    def sample_slice(self, image):
        """
        Samples a slice from the image
        :param image: A `numpy.ndarray` of the image
        """
        j = random.choice(range(self.size // 2, image.shape[-2] - self.size // 2, self.step))
        i = random.choice(range(self.size // 2, image.shape[-1] - self.size // 2, self.step))
        j, i = map(int, (j, i))
        crop_slice = (slice(None, None),
                      slice((j - self.size // 2), j + self.size // 2),
                      slice(i - self.size // 2, i + self.size // 2))
        return crop_slice

    def __len__(self):
        return len(self.valid_index)


class MultipleSampler(Sampler):
    """
    Creates a `Sampler` that samples multiple crops in an image instance
    """

    def __init__(self, dataset):
        """
        Inits `MultipleSampler`
        :param dataset: A `Dataset` instance
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.random_samples = numpy.random.randint(10, 20, size=len(self.dataset))

    def __iter__(self):
        """
        In each image we randomly sample a number of images
        """
        samples_per_image = []
        for i, samples in zip(numpy.random.permutation(len(self.dataset)), self.random_samples):
            samples_per_image.extend([i] * samples)
        random.shuffle(samples_per_image)
        return iter(samples_per_image)

    def __len__(self):
        """
        Returns the total number of samples
        """
        return sum(self.random_samples)


def HDF5Loader(file_path, **kwargs):
    """
    Creates the dataset instance and the dataloader instance
    :returns : A `DataLoader` instance
    """
    dataset = Dataset(file_path, **kwargs)
    sampler = MultipleSampler(dataset)
    dataloader_params = kwargs["hdf5_params"]
    return DataLoader(dataset=dataset,
                      sampler=sampler,
                      batch_size=kwargs["batch_size"],
                      num_workers=dataloader_params["num_workers"],
                      pin_memory=dataloader_params["pin_memory"],
                      drop_last=dataloader_params["drop_last"])


if __name__ == "__main__":

    file_path = "/home-local/Post-Synapse/training_synapse_2019-06-26.hdf5"
    trainer_params = {
        "savefolder": "/home-local/Multilabel-Proteins-Malaria/Results",
        "size": 256,
        "batch_size": 48,
        "cuda": True,
        "data_aug": 0.5,
        "step": 192,
        "adjust_pos_weight": True,
        "num_classes": 1,
        "num_input_images": 2,
        "omit_transforms": ["periodical_transform", "rotation90", "elastic_transform"],
        "hdf5_params": {
            "use": True,
            "load_data": True,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": True,
            "data_cache_size": 8e+9
        }
    }
    loader = HDF5Loader(file_path, **trainer_params)

    from tqdm import tqdm

    for x, y in tqdm(loader):
        (x.shape, y.shape)