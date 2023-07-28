# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:19:35 2023

@author: prarthana.ts
"""
from torchvision import datasets, transforms
import torch
cifar10_classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## Calculate Dataset Statistics
def return_dataset_statistics():
    
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root='./data', train = True, download = True, transform = train_transform)
    
    mean = train_set.data.mean(axis=(0,1,2))/255
    std = train_set.data.std(axis=(0,1,2))/255

    return mean, std


def return_datasets(train_transforms, test_transforms):
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform = train_transforms)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform = test_transforms)
    
    return trainset, testset

def return_dataloaders(trainset, testset, cuda, gpu_batch_size = 512, cpu_batch_size = 64):
    
    dataloader_args = dict(shuffle = True, batch_size = gpu_batch_size, num_workers = 4, pin_memory = True) if cuda else dict(shuffle = True, batch_size = cpu_batch_size)

    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
    
    return trainloader, testloader
