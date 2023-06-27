
import torch
from dataclasses import dataclass
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from PIL import Image

import os

# Data pipeline
# TO DO: add normalization in transform V
# zorgt ImageFolder voor labels en images scheiden? Voor de mapping van classnummer naar classnaam V (target_transform)
# TO DO: add data augmentation before Conv2d layer, je kan de transformed data toevoegen aan dataset. Wat ga ik doen met de transformede plaatjes? 2 datasets maken met transform en dan concatenaten.
# org_dataset, aug_dataset
# Nieuwe map plaatjes in een andere map saven met data_transformed (storage)


class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
 #       super().__init__()
        self.data_path = data_path
        self.transforms = transform
        self.img_folder = ImageFolder(self.data_path, self.transforms)
        self.images = [i[0] for i in self.img_folder]
        self.labels = [l[1] for l in self.img_folder]
        
    def __getitem__(self, index):
        
        image = self.images[index]
        label = self.labels[index]
        
        return image, label
    
    def __len__(self):
        return len(self.images)
     

