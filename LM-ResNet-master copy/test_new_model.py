import torch
import torchvision
import torchvision.transforms as transforms
from MResNet import *
import numbers
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

def write_file_and_close(filename, *arg, flag = "a"):
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
    return 'result/'+modelname + '-test' + str(code) + ".pt"

class NN_SGDTrainer(object):
    def __init__(self,net,sgd_para,trainloader,testloader,lr_adjust,device,global_output_filename = "out.txt"): # Added device
        self.net = net
        self.sgd_para = sgd_para
        self.optimizer = optim.SGD(net.parameters(), **sgd_para)
        self.trainloader = trainloader
        self.testloader = testloader
        self.output = global_output_filename
        self.lr_adjust = lr_adjust
        self.iter_time = 0
        self.max = 0
        self.device = device # New parameter

    def renew_trainer(self):
        self.optimizer = optim.SGD(self.net.parameters(), **self.sgd_para)

    def train(self,model_name="test"):
        self.iter_time += 1
        acc = train_epoch(self.net,self.optimizer,self.trainloader,self.testloader,self.iter_time,self.lr_adjust,self.device,self.output)
        if acc > self.max:
            model_filename = generate_filename(model_name, 1)
            xm.save(self.net.state_dict(), model_filename)   # xm.save instead of torch.save to go back to cpu
            self.max = acc

    def net_test(self):
        criterion = nn.CrossEntropyLoss()
        def test(info):
            (self.net).eval()  # trun to eval mode
            correct_sum = 0
            total_loss_sum = 0.
            total_ctr = 0
            for i, data in enumerate(self.testloader):
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
                if i%10==0:
                    print('batch i, total_loss_sum = ', total_loss_sum)
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

'''
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
testset = torchvision.datasets.CIFAR10(
        root="./data", download=True, train=False, transform=transform_test
    )
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)
'''
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

    testset = torchvision.datasets.CIFAR10(
        root="./data", download=True, train=False, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader,testloader

state_dict = torch.load('/media/jamadri/Lexar72/USB2021/DD Lisbonne/2021-2022/Master Thesis/Dynamical-Stochastic-ResNet/LM-ResNet-master copy/result/test-test1.pt')
model = MResNet20()
_ = model.load_state_dict(state_dict)

batch_size = 128
model_name = "Resnet20"
#inp=Variable(torch.FloatTensor(128,3,32,32).uniform_(0,1))
trainloader,testloader = get_cifar10(batch_size, device="cpu")
sgd_para = {"lr":1e-3}
Trainer = NN_SGDTrainer(model,sgd_para, trainloader, testloader, {200:1e-3}, "cpu", model_name + '.txt')  # Added device
print(model)