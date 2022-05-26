import timing
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 64

"""For the data sets CIFAR10 and CIFAR100, we train and test our networks on the training and testing set as originally given by the data set."""
TRAINING_DATA = datasets.CIFAR10(
    root="data",
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(),]
    ),
    download=True,
)

# Create data loaders.
TRAIN_DATALOADER = DataLoader(TRAINING_DATA, batch_size=BATCH_SIZE)


