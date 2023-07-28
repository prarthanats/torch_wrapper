# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:35:07 2023

@author: prarthana.ts
"""

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


train_losses = []
test_losses = []
train_acc = []
test_acc = []

## Get Learning Rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model, device, train_loader, train_acc, train_loss, optimizer, scheduler, criterion, lrs,epoch,tensorsummary):

    model.train()
    pbar = tqdm(train_loader)

    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):

        ## Get data samples
        data, target = data.to(device), target.to(device)

        ## Init
        optimizer.zero_grad()

        ## Predict
        y_pred = model(data)

        ## Calculate loss
        loss = criterion(y_pred, target)

        train_loss.append(loss.data.cpu().numpy().item())
        tensorsummary.add_scalar(
                'batch/Train/train_loss', loss.data.cpu().numpy().item(), epoch*len(pbar) + batch_idx)
        ## Backpropagation
        loss.backward()

        optimizer.step()
        scheduler.step()
        lrs.append(get_lr(optimizer))

        ## Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} LR={lrs[-1]:0.5f} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)



def test(model, device, test_loader, test_acc, test_losses, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))
    
