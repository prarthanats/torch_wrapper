# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 21:23:29 2023
@author: prarthana.ts
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch_lr_finder import LRFinder

import matplotlib.pyplot as plt
import numpy as np

from torch_wrapper.utils.data_loader import *
from torch_wrapper.utils import data_handeling
from torch_wrapper.utils.train_test import train,test


class TriggerTraining:
    def __init__(self, config):
        self.config = config
        self.loader = config['data_loader']['type']
        self.image_dataset=eval(self.loader)(self.config)
        self.tensorsummary = SummaryWriter()
        self.device = self.set_device()

    def dataloader(self):
        #Get dataloaders
        return self.image_dataset.get_dataloader()
       
    def get_classes(self): 
        return self.image_dataset.classes()
        
    def set_device(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        return device

    def model_summary(self,model, input_size):
        result = summary(model, input_size=input_size)
        print(result)    

    def learning_finder(self,model,train_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=0.01)
        lr_finder = LRFinder(model, optimizer, criterion)
        lr_finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode="exp")  # Adjust end_lr and num_iter as needed
        val,max_LR = lr_finder.plot()
        print("LRMAX:", max_LR)
        lr_finder.reset()  
        return max_LR

    def run_experiment(self,model,train_loader,test_loader,lrmax):
        train_losses = []
        test_losses = []
        train_accuracy = []
        test_accuracy = []
        plot_train_acc=[]
        lrs=[]
        
        model.to(self.device) 
        epochs=self.config['training_params']['epochs']
        max_epoch = self.config['max_lr_epoch']
        criterion = nn.CrossEntropyLoss() if self.config['criterion'] == 'CrossEntropyLoss' else F.nll_loss()
        opt_func = optim.Adam if self.config['optimizer']['type'] == 'optim.Adam' else optim.SGD

        optimizer = opt_func(model.parameters(), lr=0.01)
        if self.config['lr_scheduler'] == 'OneCycleLR':
            scheduler = OneCycleLR(optimizer, max_lr=lrmax,pct_start=max_epoch/epochs,epochs=epochs,steps_per_epoch=len(train_loader))
        else:
            scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3,verbose=True,mode='max')

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}:')
            train(model, self.device, train_loader, train_accuracy, train_losses, optimizer, scheduler, criterion, lrs,epoch,self.tensorsummary)
            test(model, self.device, test_loader, test_accuracy, test_losses, criterion)
            
            self.tensorsummary.add_scalar('epoch/Train/train_accuracy', train_accuracy[-1], epoch)
            self.tensorsummary.add_scalar('epoch/Train/train_loss', train_losses[-1], epoch)
            self.tensorsummary.add_scalar('epoch/Train/test_accuracy', test_accuracy[-1], epoch)
            self.tensorsummary.add_scalar('epoch/Test/test_loss', test_losses[-1], epoch)
            plot_train_acc.append(train_accuracy[-1])
            
            if "ReduceLROnPlateau" in str(scheduler):
                scheduler.step(test_accuracy[-1])
                
            self.tensorsummary.flush()

        return plot_train_acc,train_accuracy,train_losses,test_accuracy,test_losses

    def wrong_predictions(self,model,test_loader,num_img):
        wrong_images=[]
        wrong_label=[]
        correct_label=[]
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)        
                pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

                wrong_pred = (pred.eq(target.view_as(pred)) == False)
                wrong_images.append(data[wrong_pred])
                wrong_label.append(pred[wrong_pred])
                correct_label.append(target.view_as(pred)[wrong_pred])  
      
                wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))    
            print(f'Total wrong predictions are {len(wrong_predictions)}')
            
            self.plot_misclassified(wrong_predictions,num_img)
      
        return wrong_predictions
        
    def plot_misclassified(self,wrong_predictions,num_img):
        fig = plt.figure(figsize=(15,12))
        fig.tight_layout()
        mean,std = self.image_dataset.calculate_mean_std()
        for i, (img, pred, correct) in enumerate(wrong_predictions[:num_img]):
            img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
            for j in range(img.shape[0]):
                img[j] = (img[j]*std[j])+mean[j]
            
            img = np.transpose(img, (1, 2, 0)) 
            ax = fig.add_subplot(5, 5, i+1)
            fig.subplots_adjust(hspace=.5)
            ax.axis('off')
            self.class_names,_ = self.get_classes()
            
            ax.set_title(f'\nActual : {self.class_names[target.item()]}\nPredicted : {self.class_names[pred.item()]}',fontsize=10)  
            ax.imshow(img)  
          
        plt.show()

    
