'''
Code For 
Yiping Lu, Aoxiao Zhong, Quanzheng Li, Bin Dong. "Beyond Finite Layer Neural Network:Bridging Deep Architects and Numerical Differential Equations" 
Thirty-fifth International Conference on Machine Learning (ICML), 2018

@2prime @Aoxiao Zhong

Acknowledgement:
@pppppass(Zhihan Li)

Adapted as a python file to run on google's TPUs
'''

### SUPER IMPORTANT ###
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

###

import torch
import os
from MResNet import *
from torch.autograd import Variable
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import re
import numpy
import sys, time
import argparse
import datetime
'''
Trying to optimize memory access with sharding see https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm

def my_worker_splitter(urls):
   """Split urls per worker
   Selects a subset of urls based on Torch get_worker_info.
   Used as a shard selection function in Dataset.
   replaces wds.split_by_worker"""

   urls = [url for url in urls]

   assert isinstance(urls, list)

   worker_info = torch.utils.data.get_worker_info()
   if worker_info is not None:
       wid = worker_info.id
       num_workers = worker_info.num_workers

       return urls[wid::num_workers]
   else:
       return urls
def my_node_splitter(urls):
   """Split urls_ correctly per accelerator node
   :param urls:
   :return: slice of urls_
   """
   rank=xm.get_ordinal()
   num_replicas=xm.xrt_world_size()

   urls_this = urls[rank::num_replicas]
  
   return urls_this
# The two functions above were definetly new
'''

def write_file_and_close(filename, *arg, flag = "a"):
    if xm.is_master_ordinal():  # To only run on the main chore and not on all of them.
        with open(filename, flag) as output_file:
            output_file.write(str(datetime.datetime.now()))
            output_file.write(":\n")
            output_file.write(*arg)
            output_file.write("\n")
            print(*arg) 

def check_control(filename):
    with open(filename, "r") as filename:
        try:
            u = int(filename.read().strip())
            return bool(u)
        except:
            write_file_and_close("Error occured checking control!!!")
            return False

def generate_filename(modelname,code = None):
    # if code = None, generate tim as the code
    if code == None:
        return 'result/'+modelname + '-' + datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".pt"
    return 'result/'+modelname + str(code) + ".pt"


def train_epoch(net,optimizer,trainloader,testloader,it,control_dict,device,global_output_filename = "out.txt"):
    """
    SGD Trainer For Cross Entropy Loss
    :param net:
    :param sgd_para: sgd parameters(should be a dictionary)
    :param trainloader:
    :param testloader:
    :param it: iteration time
    :param lr_adjust: learning rate adjust
    :param global_output_filename:
    :param global_control_filename:
    :return:None
    """
    # to do L2loss or other losses

    def lr_control(control_dict, it):
        for i in control_dict:
            if it <= i:
                return control_dict[i]

    criterion = nn.CrossEntropyLoss()

    # global_cuda_available = False # torch.cuda.is_available()
    #if global_cuda_available:
    #    net = net.cuda()



    def train(data, info):
        net.train()#turn to train mode
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = inputs.to(device), labels.to(device)  # Changes here, not interested in cuda but in tpu
        # if global_cuda_available:
        #     inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)#,labels)
        loss = criterion(outputs, labels)
        loss.backward()
        # optimizer.step()
        xm.optimizer_step(optimizer) # After the local gradients are computed, the xm.optimizer_step() function synchronizes the local gradients between cores by applying an AllReduce(SUM) operation, and then calls the PyTorch optimizer_step(optimizer), which updates the local weights with the synchronized gradients.See https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm
        # print("loss.data: ",loss.data)
        info[0] = loss.data #[0]
        info[1] = labels.size()[0]

    def test(info):
        net.eval()# trun to eval mode
        correct_sum = 0
        total_loss_sum = 0.
        total_ctr = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)  # Volatile =True change to with torch.no_grad because of a pytorch update.
                inputs, labels = inputs.to(device), labels.to(device)  # Changes here interested in TPUs not cuda
                #if global_cuda_available:
                #    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)#,labels)
                _, predicted = torch.max(outputs.data, 1)
                total_ctr += labels.size()[0]
                correct_sum += (predicted == labels.data).sum()
                loss = criterion(outputs, labels)
                total_loss_sum += loss.data # [0]
        info[0] = correct_sum
        info[1] = total_ctr
        info[2] = total_loss_sum

    running_loss_sum = 0.
    total_loss_sum = 0.
    ctr_sum = 0
    total_ctr = 0
    for g in optimizer.param_groups:
        g["lr"] = lr_control(control_dict,it)
    for i, data in enumerate(trainloader):
        info = [0., 0]
        train(data, info)
        running_loss_sum += info[0]
        total_loss_sum += info[0]
        ctr_sum += 1
        total_ctr += info[1]
        if (i + 1) % 20 == 0:  # I want more prints to estimate the time it takes
            write_file_and_close(global_output_filename,
                                    "epoch: {:d}, "
                                    "train set index: {:d}, "
                                    "average loss: {:.10f}"
                                    .format(it, i, running_loss_sum / ctr_sum)
                                    )
        running_loss_sum = 0.0
        ctr_sum = 0
        # it = it + 1
    write_file_and_close(global_output_filename,
                         "Epoch {:d} finished, average loss: {:.10f}"
                         .format(it, total_loss_sum / total_ctr)
                         )
    if True: # True:  Changed to false because I don't want to run any test yet.
        write_file_and_close(global_output_filename, "Starting testing")
        info = [0., 0., 0.]
        test(info)
        write_file_and_close(global_output_filename,
                             "Correct: {:d}, total: {:d}, "
                             "accuracy: {:.10f}, average loss: {:.10f}"
                             .format(info[0], info[1], info[0] / info[1], info[2] / info[1])
                             )
        return info[0]


class NN_SGDTrainer(object):
    def __init__(self,net,sgd_para,trainloader,testloader,lr_adjust,device,global_output_filename = "out.txt", code=None): # Added device
        self.net = net
        self.sgd_para = sgd_para
        self.optimizer = optim.SGD(net.parameters(), **sgd_para)
        self.trainloader = trainloader
        self.testloader = testloader
        self.output = global_output_filename
        self.lr_adjust = lr_adjust
        self.iter_time = 0
        self.max = 0
        self.code = code
        self.device = device # New parameter

    def renew_trainer(self):
        self.optimizer = optim.SGD(self.net.parameters(), **self.sgd_para)

    def train(self,model_name="exp"):
        self.iter_time += 1
        acc = train_epoch(self.net,self.optimizer,self.trainloader,self.testloader,self.iter_time,self.lr_adjust,self.device,self.output)
        if acc > self.max:
            model_filename = generate_filename(model_name, self.code)
            xm.save(self.net.state_dict(), model_filename)   # xm.save instead of torch.save to go back to cpu
            self.max = acc
    ''' Not called
    def net_test(self): 
        criterion = nn.CrossEntropyLoss()
        def test(info):
            self.net.eval()  # trun to eval mode
            correct_sum = 0
            total_loss_sum = 0.
            total_ctr = 0
            for data in self.testloader:
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                # Change here, not interested in cuda, want to do it on TPU
                #if torch.cuda.is_available(): 
                #    inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_ctr += labels.size()[0]
                correct_sum += (predicted == labels.data).sum()
                loss = criterion(outputs, labels)
                total_loss_sum += loss.data
            info[0] = correct_sum
            info[1] = total_ctr
            info[2] = total_loss_sum

        write_file_and_close(self.output, "Starting testing")
        info = [0., 0., 0.]
        test(info)
        write_file_and_close(self.output,
                             "Correct: {:d}, total: {:d}, "
                             "accuracy: {:.10f}, average loss: {:.10f}"
                             .format(info[0], info[1], info[0] / info[1], info[2] / info[1])
                             )
    '''

    def plot_loss(self,label,filename):
        extract_loss = re.compile(r"Epoch (\d+) finished, average loss: (\d.\d+)")
        arr_loss = []
        with open(self.output) as f:
            ctr = 0
            for line in f:
                match = extract_loss.match(line)
                if match:
                    arr_loss.append(match.group(2))
        nparr_loss = numpy.array(arr_loss)
        plt.plot(nparr_loss, label=label)
        plt.legend()
        plt.savefig(filename)

    def plot_train(self,lable,filename):
        extract_accuracy = re.compile(r"Correct: (\d+), total: (\d+), accuracy: (\d.\d+), average loss: (\d.\d+)")
        arr_acc = []
        with open(self.output) as f:
            ctr = 0
            for line in f:
                match = extract_accuracy.match(line)
                if match:
                    arr_acc.append(match.group(3))
        nparr_acc = numpy.array(arr_acc)
        plt.plot(nparr_acc, label=lable)
        plt.legend()
        plt.savefig(filename)

    def get_net(self):
        return self.net

    def write(self,out_txt):
        write_file_and_close(self.output,out_txt)



import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import numbers
import functools
import numpy as np

"""
Cifar10 dataset Loading API
With data
@2prime 2017
"""


def expand_reflect(image, border=0):
    """
    Add border to the image(Symmetric padding)

    :param image: The image to expand.
    :param border: Border width, in pixels.
    :return: An image.
    """
    img = np.asarray(image)
    img = np.pad(img,pad_width=border,mode="reflect")
    return Image.fromarray(np.uint8(img[:,:,2:5]))

class Reflect_Pad(object):
    """Pads the given PIL.Image on all sides with the given "pad" reflect"""

    def __init__(self, padding):
        assert isinstance(padding, numbers.Number)
        self.padding = padding

    def __call__(self, img):
        return expand_reflect(img, border=self.padding)



def get_cifar10(batch_size, device):
    '''
    :param batch_size:
    :return: trainloader,testloader
    cifar10 dataset loader

    trainloader,testloader = get_cifar10(batch_size)
    '''
    transform_train = transforms.Compose([
        Reflect_Pad(2),
        transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", download=True, train=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    trainloader = pl.MpDeviceLoader(trainloader, device)  # See why here: http://pytorch.org/xla/release/1.11/index.html#running-on-multiple-xla-devices-with-multi-processing

    testset = torchvision.datasets.CIFAR10(
        root="./data", download=True, train=False, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    testloader = pl.MpDeviceLoader(testloader, device)  # See why here: http://pytorch.org/xla/release/1.11/index.html#running-on-multiple-xla-devices-with-multi-processing

    return trainloader,testloader

def get_cifar100(batch_size):
    '''
    :param batch_size:
    :return: trainloader,testloader
    cifar10 dataset loader

    trainloader,testloader = get_cifar10(batch_size)
    '''
    transform_train = transforms.Compose([
        Reflect_Pad(2),
        transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR100(
        root="./data", download=True, train=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR100(
        root="./data", download=True, train=False, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader,testloader





def _run():  # See https://www.kaggle.com/code/tanulsingh077/pytorch-xla-understanding-tpu-s-and-xla/notebook
    ### SUPER IMPORTANT
    dev = xm.xla_device()
    MEAN = torch.Tensor([0.4914, 0.4822, 0.4465])
    STD = torch.Tensor([0.2023, 0.1994, 0.2010])
    NORMALIZED_MIN = -MEAN/STD
    NORMALIZED_MAX = (1-MEAN)/STD
    normalized_min_clip = NORMALIZED_MIN.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(dev)
    normalized_max_clip = NORMALIZED_MAX.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(dev)

    code=100
    configPGD={
    'epsilon':0.031,
    'num_steps':10,
    'step_size':0.007, # $\alpha$ in the paper
    'random_start':True,
    'loss_func':'xent',
    'normalized_min_clip':normalized_min_clip,
    'normalized_max_clip':normalized_max_clip
    }
    # state_dict = torch.load('/result/test-test1.pt')
    '''
    In our experiments, we select pL = 0.8 for LM-ResNet56 and pL = 0.5 for LM-ResNet110.
    '''
    ResNetParameters={"block":BasicBlock,"layers":[3,3,3],"pretrain":False,"num_classes":10, "device":dev,"noise_level":None}

    '''net=MResNet(**MResNetParameters)
    net=AttackPGD(net, configPGD)
    net=En_LM_ResNet(net,num_ensembles=2)
    state_dict = torch.load('result/ResNet20Normal.pt')
    net.load_state_dict(state_dict)
    '''
    net=ResNet(**ResNetParameters)
    net.to(device=dev)
    model_name = "ResNet20Normal"
    # net.load_state_dict(state_dict)
    ###
    batch_size = 128
    #inp=Variable(torch.FloatTensor(128,3,32,32).uniform_(0,1))
    # Adding rendezvous :see here an example https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/multi-core-alexnet-fashion-mnist.ipynb?hl=id-id&skip_cache=true#scrollTo=pqx-VgEizPiF
    
    trainloader,testloader = get_cifar10(batch_size, dev)

    '''During training with SGD, the initial learning rate is 0.1, and is divided 
    by a factor of 10 after epoch 250 and 375, and terminated at 500 epochs.
    In addition, we use a weight decay of 0.0001 and a momentum of 0.9.'''
    '''
    During training, we apply a weight decay of 0.0001 for LM-ResNet and 0.0005 for LM-ResNeXt, and momentum of 0.9 on CIFAR.
    '''
    '''
    For LM-ResNet on CIFAR10 (CIFAR100), we start with the learning rate of 0.1, divide it by 10 at 80 (150) and 120 (225) epochs and terminate training at 160 (300) epochs.
    '''
    sgd_para = {"lr":0.1, "momentum":0.9, "weight_decay":0.0001}
    last_epoch_computed=160
    Trainer = NN_SGDTrainer(net,sgd_para, trainloader, testloader, {40:0.0001}, dev, model_name+'.txt', code)
    for i in range(40):
        Trainer.train()
def _mp_fn(rank, flags):
    '''
    torch.set_default_tensor_type('torch.FloatTensor')
    This is not necessary: the default type is already FloatTensor
    "The default floating point tensor type is initially torch.FloatTensor." see 
    https://pytorch.org/docs/stable/generated/torch.set_default_tensor_type.html
    '''
    a = _run()
if __name__=="__main__":
    FLAGS={}
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')