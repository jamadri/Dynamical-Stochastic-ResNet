import timing
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 1. LOADING AND AUGMENTING DATA
'''
“For the data sets CIFAR10 and CIFAR100, we train and test our networks on the training and testing set as originally given by the data set.” ([Lu et al., 2020, p. 5])
“On CIFAR, we follow the simple data augmentation in Lee et al. (2015) for training: 4 pixels are padded on each side, and a 32x32 crop is randomly sampled from the padded image or its horizontal flip.” ([Lu et al., 2020, p. 5])
'''
TRAINING_DATA = datasets.CIFAR10(
    root="data",
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(),
        transforms.Pad(4), # "4 pixels are padded on each side"
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32), # "and a 32×32 crop is randomly sampled from the padded image or its horizontal flip."
        ]
    ),
    download=True,
)
TESTING_DATA = datasets.CIFAR10(
    root="data",
    train=False,
    transform = transforms.ToTensor()
)

# 2. CREATE DATALOADER ITERATORS.
BATCH_SIZE = 128  # “On CIFAR, we use SGD with a mini-batch size of 128” ([Lu et al., 2020, p. 5])


TRAIN_DATALOADER = DataLoader(TRAINING_DATA, batch_size=BATCH_SIZE) # [N,C,H,W] = [64,3,32,32]
TEST_DATALOADER = DataLoader(TESTING_DATA, batch_size=BACTH_SIZE)

# 3. SET THE MODEL, LOSS AND OPTIMIZER
MODEL = LM_resNet(n=1, num_classes=10)  # We should have 6n blocks if everything is correct:  “Then we use a stack of 6n layers” ([He et al., 2015, p. 7])
LOSS_FN = nn.CrossEntropyLoss()  # As in the original training code https://github.com/2prime/LM-ResNet/blob/master/cifar10_training.ipynb (search for "criterion")
OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=?)








