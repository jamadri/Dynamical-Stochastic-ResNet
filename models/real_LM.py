import torch
import torch.nn as nn
import torch.nn.functional as functional
import math
from torch.autograd import Variable

from init import *
from random import random

class BasicBlockWithDeathRate(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,death_rate=0., downsample=None):
        super(BasicBlockWithDeathRate,self).__init__()
        self.bn1=nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,padding=1,bias=False)
        self.relu=nn.ReLU(inplace=True) 
        self.stride=stride
        self.in_planes=in_planes
        self.planes=planes
        self.death_rate=death_rate
    def forward(self,x):
        if not self.training or torch.rand(1)[0] >= self.death_rate:
            out=self.bn1(x)
            out=self.relu(out)
            out=self.conv1(out)
            out=self.bn2(out)
            out=self.relu(out)
            out=self.conv2(out)
            if self.training:
                out /= (1. - self.death_rate)  # During training phase some of the channels are ignored, we rescale the outputs as a consequence.
        else:  # We are in training phase but do not kill any channel.
            if self.stride==1:  # 
                out=Variable(torch.FloatTensor(x.size()).cuda().zero_(),requires_grad=False)
            else:
                size=list(x.size())
                size[-1]//=2
                size[-2]//=2
                size[-3]*=2
                size=torch.Size(size)
                out=Variable(torch.FloatTensor(size).cuda().zero_(),requires_grad=False)
        return out    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock,self).__init__()
        self.bn1=nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,padding=1,bias=False)
        self.relu=nn.ReLU(inplace=True) 
        self.stride=stride
        self.in_planes=in_planes
        self.planes=planes
    def forward(self,x):
        out=self.bn1(x)
        out=self.relu(out)
        out=self.conv1(out)
        out=self.bn2(out)
        out=self.relu(out)
        out=self.conv2(out)

        return out
    

class GaussianNoise(nn.Module):  # Not sure when is it used?
    def __init__(self, stddev):
        super(GaussianNoise,self).__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            return x + torch.autograd.Variable(torch.randn(x.size()).cuda() * self.stddev,requires_grad=False)
        return x
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__( self, in_planes, planes, stride=1):
        super(Bottleneck,self).__init__()
        self.bn1=nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn3=nn.BatchNorm2d(planes)
        self.conv3=nn.Conv2d(planes,planes*4,kernel_size=1,bias=False)
        self.relu=nn.ReLU(inplace=True)

        self.in_planes = in_planes
        self.planes=planes
    def forward(self,x):
        

        out=self.bn1(x)
        out=self.relu(out)
        out=self.conv1(out)
        
        out=self.bn2(out)
        out=self.relu(out)
        out=self.conv2(out)
        
        out=self.bn3(out)
        out=self.relu(out)
        out=self.conv3(out)

        return out

class Downsample(nn.Module):  # “We perform downsampling directly by convolutional layers that have a stride of 2.” ([He et al., 2015, p. 3]
    def __init__(self,in_planes,out_planes,stride=2):
        super(Downsample,self).__init__()
        self.downsample=nn.Sequential(
                        nn.BatchNorm2d(in_planes),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_planes, out_planes,
                                kernel_size=1, stride=stride, bias=False)
                        )
    def forward(self,x):
        x=self.downsample(x)
        return x

class MResNet(nn.Module):
    # “on CIFAR, we adopt the original design of the residual block in He et al. (2016), i.e. using a small two-layer neural network as the residual block with bn-relu-conv-bn-reluconv.” ([Lu et al., 2020, p. 5])
    def __init__(self,block,layers,pretrain=True,num_classes=100,stochastic_depth=False,PL=0.5,noise_level=0.001,noise=False):
        self.in_planes=16
        self.planes=[16,32,64]
        self.strides=[1,2,2]
        super(MResNet,self).__init__()
        self.noise=noise 
        self.block=block
        self.conv1=nn.Conv2d(3,16,kernel_size=3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.relu=nn.ReLU(inplace=True)
        self.pretrain=pretrain
        self.ks=nn.ParameterList([nn.Parameter(torch.Tensor(1).uniform_(1.0, 1.1))for i in range(layers[0]+layers[1]+layers[2])])
        self.stochastic_depth=stochastic_depth
        blocks=[]
        n=layers[0]+layers[1]+layers[2]
        
        if not self.stochastic_depth:
            for i in range(3):
                blocks.append(block(self.in_planes,self.planes[i],self.strides[i]))
                self.in_planes=self.planes[i]*block.expansion
                for j in range(1,layers[i]):
                    blocks.append(block(self.in_planes,self.planes[i]))
        else:
            death_rates=[i/(n-1)*(1-PL) for i in range(n)]
            print(death_rates)
            for i in range(3):
                blocks.append(block(self.in_planes,self.planes[i],self.strides[i],death_rate=death_rates[i*layers[0]]))
                self.in_planes=self.planes[i]*block.expansion
                for j in range(1,layers[i]):
                    blocks.append(block(self.in_planes,self.planes[i],death_rate=death_rates[i*layers[0]+j]))
        self.blocks=nn.ModuleList(blocks)
        self.downsample1=Downsample(16,64,stride=1)
        #self.downsample1=nn.Conv2d(16, 64,
        #                    kernel_size=1, stride=1, bias=False)
        self.downsample21=Downsample(16*block.expansion,32*block.expansion)
        #self.downsample22=Downsample(16*block.expansion,32*block.expansion)
        self.downsample31=Downsample(32*block.expansion,64*block.expansion)
        #self.downsample32=Downsample(32*block.expansion,64*block.expansion)

        self.bn=nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def change_state(self):
        self.pretrain=not self.pretrain
    


    def forward(self,x):
        '''
        We obtain the following stochastic training strategy for LM-architecture
        Xn+1 = (2 + gn)Xn - (1 + gn)Xn-1 + ηn f(Xn).
        '''
        x=self.conv1(x)
        #x=self.bn1(x)
        #x=self.relu(x)
        
        if self.block.expansion==4:
            residual=self.downsample1(x)
        else:
            residual=x
        x=self.blocks[0](x)+residual
        last_residual=residual
        for i,b in enumerate(self.blocks):
            if i==0:
                continue
            residual=x
                
            if b.in_planes != b.planes * b.expansion :
                if b.planes==32:
                    residual=self.downsample21(x)
                    #if not self.pretrain:
                        #last_residual=self.downsample22(last_residual)
                elif b.planes==64:
                    residual=self.downsample31(x)
                    #if not self.pretrain:
                        #last_residual=self.downsample32(last_residual)
                x=b(x)
                #print(x.size())
                #print(residual.size()) 
                x+=residual
                
            elif self.pretrain:
                x=b(x)+residual                
            else:
                x=b(x)+self.ks[i].expand_as(residual)*residual+(1-self.ks[i]).expand_as(last_residual)*last_residual
            
            last_residual=residual
        
        x=self.bn(x)
        x=self.relu(x)
        x=self.avgpool(x)
        x=x.view(x.size(0), -1)
        x=self.fc(x) 
        return x


class MResNetC(nn.Module):

    def __init__(self,block,layers,pretrain=False,num_classes=100):
        self.in_planes=16
        self.planes=[16,32,64]
        self.strides=[1,2,2]
        super(MResNetC,self).__init__()
        self.block=block
        self.conv1=nn.Conv2d(3,16,kernel_size=3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.relu=nn.ReLU(inplace=True)
        self.pretrain=pretrain
        ks=[]
        for i in range(3):
            ks+=[nn.Parameter(torch.Tensor(self.planes[i]*block.expansion,1,1).uniform_(-0.1, -0.0)) for j in range(layers[i])]
        self.ks=nn.ParameterList(ks)
        
        blocks=[]
        for i in range(3):
            blocks.append(block(self.in_planes,self.planes[i],self.strides[i]))
            self.in_planes=self.planes[i]*block.expansion
            for j in range(1,layers[i]):
                blocks.append(block(self.in_planes,self.planes[i]))
        self.blocks=nn.ModuleList(blocks)
        self.downsample1=Downsample(16,64,stride=1)
        #self.downsample1=nn.Conv2d(16, 64,
        #                    kernel_size=1, stride=1, bias=False)
        self.downsample21=Downsample(16*block.expansion,32*block.expansion)
        #self.downsample22=Downsample(16*block.expansion,32*block.expansion)
        self.downsample31=Downsample(32*block.expansion,64*block.expansion)
        #self.downsample32=Downsample(32*block.expansion,64*block.expansion)

        self.bn=nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def change_state(self):
        self.pretrain=not self.pretrain
    


    def forward(self,x):
        x=self.conv1(x)
        #x=self.bn1(x)
        #x=self.relu(x)
        
        if self.block.expansion==4:
            residual=self.downsample1(x)
        else:
            residual=x
        x=self.blocks[0](x)+residual
        last_residual=residual
        if self.training and self.noise:
                x+=Variable(torch.FloatTensor(x.size()).cuda().uniform_(0,self.noise_level),requires_grad=False)
        for i,b in enumerate(self.blocks):
            if i==0:
                continue
            residual=x
            
            if b.in_planes != b.planes * b.expansion :
                if b.planes==32:
                    residual=self.downsample21(x)
                    #if not self.pretrain:
                        #last_residual=self.downsample22(last_residual)
                elif b.planes==64:
                    residual=self.downsample31(x)
                    #if not self.pretrain:
                        #last_residual=self.downsample32(last_residual)
                x=b(x)+residual
            elif self.pretrain:
                x=b(x)+residual                
            else:
                
                x=b(x)+self.ks[i].expand_as(residual)*residual+(1-self.ks[i]).expand_as(last_residual)*last_residual 
            last_residual=residual
            if self.training and self.noise:
                x+=Variable(torch.FloatTensor(x.size()).cuda().uniform_(0,self.noise_level),requires_grad=False)
        x=self.bn(x)
        x=self.relu(x)
        x=self.avgpool(x)
        x=x.view(x.size(0), -1)
        x=self.fc(x) 
        return x
class DenseBlock(nn.Module):
    def __init__(self,block,layers,in_planes,planes,stride=2,pretrain=False):
        super(DenseBlock,self).__init__()
        self.in_planes=in_planes
        self.planes=planes
        self.stride=stride
        self.layers=layers
        blocks=[]
        blocks.append(block(self.in_planes,self.planes,self.stride))
        for j in range(1,layers):
            blocks.append(block(self.planes,self.planes))
        
        self.downsample=None
        if in_planes!=planes*block.expansion or stride != 1:
            self.downsample=Downsample(in_planes,planes,stride=stride)
        
        self.ks=(nn.ParameterList([nn.Parameter(torch.Tensor(1).uniform_(-0.1, -0.0))for i in range(layers*layers)]))
        self.blocks=nn.ModuleList(blocks)    
    def forward(self,x):
        residuals=[]
        for i,b in enumerate(self.blocks):
            if i==0 and self.downsample!=None:
                residuals.append(self.downsample(x))
            else:
                residuals.append(x)
            
            residual=(self.ks[i*self.layers+i]).expand_as(residuals[i])*residuals[i]
            sumk=self.ks[i*self.layers+i].clone()
            for j in range(i):
                residual+=(self.ks[i*self.layers+j]).expand_as(residuals[j])*residuals[j]
                sumk+=self.ks[i*self.layers+j]
            x=residual/sumk.expand_as(residual)+b(x)
        return x
            
class DenseResNet(nn.Module):

    def __init__(self,block,layers,pretrain=False,num_classes=100):
        self.in_planes=16
        self.planes=[16,32,64]
        self.strides=[1,2,2]
        super(DenseResNet,self).__init__()
        self.block=block
        self.conv1=nn.Conv2d(3,16,kernel_size=3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.relu=nn.ReLU(inplace=True)
        self.pretrain=pretrain
        
        self.denseblock1=DenseBlock(self.block,layers[0],16,self.planes[0],1)
        
        
        self.denseblock2=DenseBlock(self.block,layers[1],self.planes[0],self.planes[1],2)
        
        
        self.denseblock3=DenseBlock(self.block,layers[2],self.planes[1],self.planes[2],2)
        
        
        
        

        self.bn=nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
               
    def change_state(self):
        self.pretrain=not self.pretrain
    


    def forward(self,x):
        x=self.conv1(x)
        #x=self.bn1(x)
        #x=self.relu(x)
        
        x=self.denseblock1(x)
        x=self.denseblock2(x)
        x=self.denseblock3(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.avgpool(x)
        x=x.view(x.size(0), -1)
        x=self.fc(x) 
        return x             


class ResNet_N(nn.Module):

    def __init__(self,block,layers,noise_level=0.001,pretrain=True,num_classes=100):
        self.in_planes=16
        self.planes=[16,32,64]
        self.strides=[1,2,2]
        super(ResNet_N,self).__init__()
        self.noise_level=noise_level
        self.block=block
        self.conv1=nn.Conv2d(3,16,kernel_size=3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.relu=nn.ReLU(inplace=True)
        self.pretrain=pretrain
        
        blocks=[]
        for i in range(3):
            blocks.append(block(self.in_planes,self.planes[i],self.strides[i]))
            self.in_planes=self.planes[i]*block.expansion
            for j in range(1,layers[i]):
                blocks.append(block(self.in_planes,self.planes[i]))
        self.blocks=nn.ModuleList(blocks)
        self.downsample1=Downsample(16,64,stride=1)
        #self.downsample1=nn.Conv2d(16, 64,
        #                    kernel_size=1, stride=1, bias=False)
        self.downsample21=Downsample(16*block.expansion,32*block.expansion)
        self.downsample22=Downsample(16*block.expansion,32*block.expansion)
        self.downsample31=Downsample(32*block.expansion,64*block.expansion)
        self.downsample32=Downsample(32*block.expansion,64*block.expansion)

        self.bn=nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def change_state(self):
        self.pretrain=not self.pretrain
    


    def forward(self,x):
        x=self.conv1(x)
        #x=self.bn1(x)
        #x=self.relu(x)
        
        if self.block.expansion==4:
            residual=self.downsample1(x)
        else:
            residual=x
        
        x=self.blocks[0](x)+residual
        if self.training:
            x+=Variable(torch.FloatTensor(x.size()).cuda().normal_(0,self.noise_level),requires_grad=False) 
        for i,b in enumerate(self.blocks):
            if i==0:
                continue
            residual=x
            
            if b.in_planes != b.planes * b.expansion :
                if b.planes==32:
                    residual=self.downsample21(x)
                    
                elif b.planes==64:
                    residual=self.downsample31(x)
                    
            
            
            x=b(x)+residual               
           
            if self.training:
                x+=Variable(torch.FloatTensor(x.size()).cuda().uniform_(0,self.noise_level),requires_grad=False) 
            
        
        x=self.bn(x)
        x=self.relu(x)
        x=self.avgpool(x)
        x=x.view(x.size(0), -1)
        x=self.fc(x) 
        return x    
    

class ResNet(nn.Module):

    def __init__(self,block,layers,noise_level=0.001,pretrain=True,num_classes=100):
        self.in_planes=16
        self.planes=[16,32,64]
        self.strides=[1,2,2]
        super(ResNet,self).__init__()
        self.noise_level=noise_level
        self.block=block
        self.conv1=nn.Conv2d(3,16,kernel_size=3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.relu=nn.ReLU(inplace=True)
        self.pretrain=pretrain
        
        blocks=[]
        for i in range(3):
            blocks.append(block(self.in_planes,self.planes[i],self.strides[i]))
            self.in_planes=self.planes[i]*block.expansion
            for j in range(1,layers[i]):
                blocks.append(block(self.in_planes,self.planes[i]))
        self.blocks=nn.ModuleList(blocks)
        self.downsample1=Downsample(16,64,stride=1)
        #self.downsample1=nn.Conv2d(16, 64,
        #                    kernel_size=1, stride=1, bias=False)
        self.downsample21=Downsample(16*block.expansion,32*block.expansion)
        self.downsample22=Downsample(16*block.expansion,32*block.expansion)
        self.downsample31=Downsample(32*block.expansion,64*block.expansion)
        self.downsample32=Downsample(32*block.expansion,64*block.expansion)

        self.bn=nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def change_state(self):
        self.pretrain=not self.pretrain
    


    def forward(self,x):
        x=self.conv1(x)
        #x=self.bn1(x)
        #x=self.relu(x)
        
        if self.block.expansion==4:
            residual=self.downsample1(x)
        else:
            residual=x
        
        x=self.blocks[0](x)+residual
        for i,b in enumerate(self.blocks):
            if i==0:
                continue
            residual=x
            
            if b.in_planes != b.planes * b.expansion :
                if b.planes==32:
                    residual=self.downsample21(x)
                    
                elif b.planes==64:
                    residual=self.downsample31(x)
                    
            
            
            x=b(x)+residual               
            
        
        x=self.bn(x)
        x=self.relu(x)
        x=self.avgpool(x)
        x=x.view(x.size(0), -1)
        x=self.fc(x) 
        return x    