import torch
import torch.nn as nn
import math
from torch.autograd import Variable

from typing import Type, Any, Callable, Union, List, Optional


def conv3x3(in_channels, out_channels, stride=1):
    """
    ALl the convolutions are supposed to be with kernel 3x3 and unbiased. Building this function avoids inconsistencies, is straightforward to read and saves coding time.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class standard_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, noise_coef=None):
        
        super(standard_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True) # Harmless memory saving , see: https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948/8
        self.conv1 = conv3x3(in_channels, out_channels, stride) # The stride might have to be 2 to reduce dimensionnality between layers.
        self.bn2 = nn.BatchNorm2d(out_channels)        
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
    def forward(self, x):
        '''
        “on CIFAR, we adopt the original design of the residual block in He et al. (2016), i.e. using a small two-layer neural network as the residual block with bn-relu-conv-bn-reluconv.” ([Lu et al., 2020, p. 5])
        '''
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out  # No residual addition, this will be made at order 2 in LM_Block

class Downsample(nn.Module):  # “We perform downsampling directly by convolutional layers that have a stride of 2.” ([He et al., 2015, p. 3]
    '''
    This is copy-pasted from the original code https://github.com/2prime/LM-ResNet/blob/master/MResNet.py
    '''
    def __init__(self,in_channels,out_channels,stride=2):
        super(Downsample,self).__init__()
        self.downsample=nn.Sequential(
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels, out_channels,
                                kernel_size=1, stride=stride, bias=False)
                        )
    def forward(self,x):
        x=self.downsample(x)
        return x

class LM_Block(nn.Module):
    '''
    Compared to the standard ResNet, the first change is blocks combine information at rank n and n-1, following a LM scheme famous for approximating efficiently solutions of ODEs.
    The second change is we add noise (diffusion) in the networks, hoping it would.
    '''
    def __init__(self, in_channels, out_channels, k, stride=1, downsample=None, noise_coef=None):
        super(LM_Block, self).__init__()
        self.block = standard_Block(in_channels, out_channels)

    def forward(self, current_residual, previous_residual):
        new_residual = self.k*former_residual + self.k*current_residual + standard_Block(current_residual) # x_{n+1}=k_n x_{n-1} + (1-k_n) x_n
        return new_residual


class LM_resNet(nn.Module):
    def __init__(self, n, num_classes=10):
        super(LM_resNet, self).__init__()
        self.k = nn.Parameter(data=Tensor([1]))  # Consider a better initialization!
        '''
        “The following table summarizes the architecture:
        output map size | 32x32 | 16x16 | 8x8
         # layers       | 1+2n  | 2n    | 2n 
         # filters      | 16    | 32    | 64
        "
        ([He et al.,2015, p. 7])
        '''
        self.conv1 = conv3x3(3, 16)
        self.standard_Block1 = standard_Block(16, 16)
        self.LM_Block1 = LM_Block(16, 16, k)
        self.LM_DownSampling1 = Downsample(16, 32)

        self.standard_Block2 = standard_Block(32, 32)
        self.LM_Block2 = LM_Block(32, 32, k)
        self.DownSampling2 = Downsample(32,64)

        self.standard_Block3 = standard_Block(64, 64)
        self.LM_Block3 = LM_Block(64, 64, k)
        
        self.layer1 = self._make_layer(2*n, 16, 16, self.k)  # layers, # filters
        self.layer2 = self._make_layer(2*n, 32, 32, self.k)
        self.layer3 = self._make_layer(2*n, 64, 64, self.k)
        self.pooling = nn.AvgPool2d(64)  # CHECK IF IT IS THE RIGHT FUNCTION; MAYBE ADAPTIVEAVGPOOLING SEE resnet torch.py
        self.fc = nn.Linear(64, num_classes, bias=False)

    def forward(self, x):
        '''
        We have to make a decision on how to handle change of dimensions with the LM updates. Indeed
        '''
        previous_residual = x.conv1(16,16)
        current_residual = self.standard_Block1(previous_residual) + previous_residual

        # Layer 1 (2n-1 (-1 because we already computed a block as a standard ResNet block to setup x_n and x_{n-1}) blocks ("layers in the original publication") with 16 channels (or filters) each. No downsampling (images are kept 32x32)
        for _ in range(2*n-1):
            new_residual = self.LM_Block1(current_residual, previous_residual)
            previous_residual = current_residual
            current_residual = new_residual

        # Transition to layer 2: Downsampling and construction of the first residue.
        previous_residual = self.LM_DownSampling1(new_residual)
        current_residual = self.standard_Block2(previous_residual)
        
        # Layer 2 (2n-1 (-1 for the same reason as above) blocks with 32 channels each. Downsampling at the beginning)
        for _ in range(2*n-1):
            new_residual=self.LM_Block2(current_residual, previous_residual)
            previous_residual = current_residual
            current_residual = new_residual

        # Transition to layer 3: Downsampling and construction of the second residue.
        previous_residual = self.LM_DownSampling2(new_residual)
        current_residual = self.standard_Block3(previous_residual)

        # Layer 3 (2n-1 blocks with 64 channels each. Downsampling at the beginning)
        for _ in range(2*n-1):
            new_residual=self.LM_Block2(current_residual, previous_residual)
            previous_residual = current_residual
            current_residual = new_residual
        out = self.pooling(new_residual)
        out = self.fc(out)
        
        return out