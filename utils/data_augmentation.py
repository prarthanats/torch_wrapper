# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:33:57 2023
@author: prarthana.ts
"""
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2

class CIFAR10Albumentation:
    
    def __init__(self):
        pass
    
    def train_transform(self,mean,std):
        train_transforms = A.Compose([
                                A.Normalize(mean = mean, std = std, always_apply = True),
                                A.PadIfNeeded(min_height=36, min_width=36, border_mode=cv2.BORDER_REFLECT),
                                A.RandomCrop(height=32, width=32),
                                A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=mean),
                                A.ToGray(p = 0.15),
                                ToTensorV2()
                                    ])
        return lambda img:train_transforms(image=np.array(img))["image"]
                                
    def test_transform(self,mean,std):
        test_transforms = A.Compose([A.Normalize(mean=mean, std=std, always_apply=True),
                                 ToTensorV2()])
        return lambda img:test_transforms(image=np.array(img))["image"]
