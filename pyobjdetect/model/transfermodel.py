import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

from pyobjdetect.utils import misc


cudnn.benchmark = True

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_data_dir():
    return os.path.join(misc.get_data_dir(), "hymenoptera_data")


def get_transforms():
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        ),
    }

    return data_transforms


def get_datasets():
    data_dir = get_data_dir()
    data_transforms = get_transforms()

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val"]}

    return image_datasets


def run(**kwargs):
    from pyobjdetect.utils import logutils

    logutils.VERBOSE = logutils.DEBUG

    bsize = kwargs.get("batch_size", 4)
    shuffle = kwargs.get("shuffle", True)
    num_workers = kwargs.get("num_workers", 4)
    gpu_num = kwargs.get("gpu_num", 0)

    image_datasets = get_datasets()
    dataset_keys = [k for k in image_datasets.keys()]
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bsize, shuffle=shuffle, num_workers=num_workers)
        for x in dataset_keys
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in dataset_keys}
    logutils.debug(f"dataset size: {dataset_sizes}")

    class_names = image_datasets[dataset_keys[0]].classes
    logutils.debug(f"classes in dataset: {class_names}")

    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
    logutils.debug(f"Running on {device}")


if __name__ == "__main__":
    run()
