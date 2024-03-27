from __future__ import print_function, division
import os, glob
import torch
import tifffile
import numpy
import matplotlib.pyplot as plt
import tqdm
import h5py

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import filters, transform

class ImageDataset(Dataset):
    """
    A custom dataset class for loading images from a directory.

    Args:
        root_dir (str): The root directory containing the image files.
        transform (callable, optional): A function/transform to apply to the image.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.files = glob.glob(os.path.join(self.root_dir, "*"))

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The total number of images.

        """
        return len(self.files)

    def __getitem__(self, item):
        """
        Returns the image at the specified index.

        Args:
            item (int): The index of the image to retrieve.

        Returns:
            dict: A dictionary containing the image.

        """
        img_name = self.files[item]
        image = numpy.load(img_name)["arr_0"]
        sample = {'image': image}

        if self.transform:
            sample = {'image': self.transform(sample['image'])}

        return sample

class CompleteImageDataset(Dataset):
    """
    Creates a `Dataset` that allows to extract crops from complete images.
    """
    def __init__(self, root_dir, size=224, step=0.75, chan=0, transform=None,
                    preload_cache=True, use_foreground=False, scale_factor=1.,
                    *args, **kwargs):
        """
        Instantiates a `CompleteImageDataset`.

        Args:
            root_dir (str): The root directory containing the images.
            size (int): The size of the crops to extract from the images.
            step (float): The step size for sliding window extraction.
            chan (int): The channel index to extract from multi-channel images.
            transform (callable): Optional transform to be applied to the image crops.
            preload_cache (bool): Whether to preload the image cache.
            use_foreground (bool): Whether to use foreground extraction.
            scale_factor (float): The scale factor to rescale the images.
        """
        self.root_dir = root_dir
        self.size = size
        self.step = step
        self.preload_cache = preload_cache
        self.chan = chan
        self.transform = transform
        self.use_foreground = use_foreground
        self.scale_factor = scale_factor

        self.data_info = []
        self.cache = {}

        self.get_data_info()

    def get_data_info(self):
        """
        Gets all images from the root directory and extracts the image crops.
        """
        image_names = glob.glob(os.path.join(self.root_dir, "**/*.tif*"), recursive=True)

        for image_name in tqdm.tqdm(image_names, desc="Images"):
            image = tifffile.imread(image_name)
            if image.ndim == 2:
                image = image[numpy.newaxis]

            # image = numpy.pad(
            #     image,
            #     ((0, 0), (self.size, self.size), )
            # )

            # Rescales from 15nm pixels to 20nm pixels
            if self.scale_factor != 1:
                image = transform.rescale(image, self.scale_factor, preserve_range=True).astype(numpy.uint16)

            if self.use_foreground:
                foreground = numpy.sum(image, axis=0)
                foreground = filters.gaussian(foreground, sigma=5)
                threshold = filters.threshold_otsu(foreground)
                foreground = foreground > threshold
            else:
                foreground = numpy.ones(image.shape[-2:], dtype=bool)

            for j in range(0, max(1, image.shape[-2] - self.size), int(self.step * self.size)):
                for i in range(0, max(1, image.shape[-1] - self.size), int(self.step * self.size)):

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
        """
        Retrieves the image and its corresponding information at the specified index.

        Args:
            index (int): The index of the image crop to retrieve.

        Returns:
            tuple: A tuple containing the information dictionary and the image crop.
        """
        info = self.data_info[index]
        if info["key"] in self.cache:
            image = self.cache[info["key"]]
            slc = info["slc"]
        else:
            image_name = info["key"]
            slc = info["slc"]
            image = tifffile.imread(image_name)
            if image.ndim == 2:
                image = image[numpy.newaxis]

            # Rescales from 15nm pixels to 20nm pixels
            if self.scale_factor != 1:
                image = transform.rescale(image, self.scale_factor, preserve_range=True).astype(numpy.uint16)

            # Clears cache and updates with most recent image
            self.cache = {}
            self.cache[image_name] = image
        return info, image[slc]

    def __getitem__(self, index):
        """
        Returns the image crop and its associated information at the specified index.

        Args:
            index (int): The index of the image crop to retrieve.

        Returns:
            dict: A dictionary containing the image crop and its associated information.
        """
        info, image = self.get_data(index)

        image = (image - image.min()) / (image.max() - image.min())
        image = image.astype(numpy.float32)

        sample = {'image': image}
        if self.transform:
            sample = {'image': self.transform(sample['image'])}
        sample["info"] = {
            key : value for key, value in info.items() if key not in ["slc"]
        }
        return sample

    def __len__(self):
        """
        Returns the total number of image crops in the dataset.

        Returns:
            int: The total number of image crops in the dataset.
        """
        return len(self.data_info)

class HDF5CompleteImageDataset(Dataset):
    """
    A dataset class for loading and processing HDF5 image data.
    """
    def __init__(self, file_path, size=256, step=0.75, *args, **kwargs):
        """
        Initializes the HDF5CompleteImageDataset.

        Args:
            file_path (str): The path to the HDF5 file.
            size (int, optional): The size of the image crops. Defaults to 256.
            step (float, optional): The step size for generating crops. Defaults to 0.75.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.file_path = file_path
        self.size = size
        self.step = step
        self.cache = {}
        self.samples = self.generate_valid_samples()

    def generate_valid_samples(self):
        """
        Generates a list of valid samples from the dataset. This is performed only
        once at each training.
        """
        samples = []
        with h5py.File(self.file_path, "r") as file:
            for group_name, group in tqdm.tqdm(file.items(), desc="Groups", leave=False):
                data = group["data"][()].astype(numpy.float32)  # Images
                label = group["label"][()]  # shape is Rings, Fibers, and Dendrite
                shapes = group["label"].attrs["shapes"]  # Not all images have same shape
                for k, (dendrite_mask, shape) in enumerate(zip(label[:, -1], shapes)):
                    for j in range(0, shape[0] - self.size, int(self.size * self.step)):
                        for i in range(0, shape[1] - self.size, int(self.size * self.step)):
                            dendrite = dendrite_mask[j: j + self.size, i: i + self.size]
                            if dendrite.sum() >= 0.1 * self.size * self.size:  # dendrite is at least 1% of image
                                samples.append((group_name, k, j, i))
                self.cache[group_name] = {"data": data, "label": label[:, :-1]}
        return samples

    def get_data(self, _type, index):
        """
        Retrieves data from the dataset.

        Args:
            _type (str): The type of data to retrieve.
            index (int): The index of the sample.

        Returns:
            numpy.ndarray: The retrieved data.
        """
        group_name, k, j, i = self.samples[index]
        image = self.cache[group_name][_type][k]
        image = (image - image.min()) / (image.max() - image.min())
        crop = image[j: j + self.size, i: i + self.size]
        return crop

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        Args:
            index (int): The index of the item.

        Returns:
            dict: A dictionary containing the image crop and additional information.
        """
        group_name, k, j, i = self.samples[index]
        crop = self.get_data("data", index)
        return {"image": crop, "info": {"sample": [group_name, k, j, i], "key": f"{self.file_path}/{group_name}-{k}-{j}-{i}.tif"}}

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.samples)

def get_image_dataset(path, *args, **kwargs):
    """
    Returns an image dataset based on the given path.

    Args:
        path (str): The path to the image dataset.

    Returns:
        ImageDataset: An instance of the appropriate image dataset class based on the path.
    """
    if os.path.isdir(path):
        return CompleteImageDataset(path, *args, **kwargs)
    elif path.endswith(".hdf5"):
        return HDF5CompleteImageDataset(path, *args, **kwargs)
    else:
        print("Image path is not defined...")
        exit()
