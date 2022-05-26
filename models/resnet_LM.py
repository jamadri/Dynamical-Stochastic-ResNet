import torch
import torch.nn as nn
import math
from torch.autograd import Variable

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
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)        
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
    def forward(self, x):
        '''
        "We adopt the original design of the residual block in He et al. (2016), i.e. using a small two-layer neural network as the residual block with bn-relu-conv-bn-reluconv."
        '''
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

class LM_Block(nn.Module):
    '''
    Compared to the standard ResNet, the first change is blocks combine information at rank n and n-1, following a LM scheme famous for approximating efficiently solutions of ODEs.
    The second change is we add noise (diffusion) in the networks, hoping it would.
    '''
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, noise_coef=None):
        
        super(LM_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True) # Harmless memory saving , see: https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948/8
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)        
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
    def forward(self, x):
        '''
        See standard_Block for justification
        '''
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += (1-k)*identity+k*





class LM_resNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, noiseCoef=None):
        super(LM_resNet, self).__init__()
    def forward(self, x):

