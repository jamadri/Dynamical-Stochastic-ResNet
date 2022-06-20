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
import numpy as np
import sys, time
import argparse
import datetime
import torch.nn.functional as F

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



def get_cifar10(batch_size):
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

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []
    i=0
    # Loop over all examples in test set
    for data, target in test_loader:
        i+=1
        print(i)

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.cross_entropy(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

if __name__ == '__main__':
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    device="gpu"

    """
    Load the trained DNN, and attack the DNN, finally save the adversarial images
    """
    # Load the model
    state_dict = torch.load('/media/jamadri/Lexar72/USB2021/DD Lisbonne/2021-2022/Master Thesis/Dynamical-Stochastic-ResNet/LM-ResNet-master copy/result/exp3.pt')
    MResNetParameters={"block":BasicBlock,"layers":[3,3,3],"pretrain":False,"num_classes":10,"stochastic_depth":False,"PL":0.8}
    net = MResNet(**MResNetParameters)
    net.to(device)
    net.load_state_dict(state_dict)
    net.eval()
    _,test_loader=get_cifar10(batch_size=1)
    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(net, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
        
    