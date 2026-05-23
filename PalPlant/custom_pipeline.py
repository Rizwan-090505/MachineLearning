import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Config vars
IMG_SZ=224
BATCH_SZ=16


base_dir = "/kaggle/input"

for root,dirs,files in os.walk(base_dir):
    if 'train' in dirs and 'valid' in dirs:
        base_dir=root
        break

train_dir = os.path.join(base_dir,"train")
valid_dir = os.path.join(base_dir,"valid")


data_transforms = {
    'train':transforms.Compose(
        [transforms.Resize(IMG_SZ,IMG_SZ),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(10),
         transforms.Normalize(0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    'valid':transforms.Compose(
            [
                transforms.Resize(IMG_SZ,IMG_SZ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
}


