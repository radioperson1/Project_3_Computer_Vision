# imports 

import torch
import torch.nn as nn

# Build pipeline
# TO DO: Van train validatie set halen. (datasplitting, hoeveel van de train?) Data splitting
# TODO Eigen train methode?
# zorgt ImageFolder voor labels en images scheiden? Voor de mapping van classnummer naar classnaam
# Hyperparameters checken en evt tweaken
# Confusion matrix: als ie fout is wat is er dan fout. 
# Model pickelen (storage)
# AQL ontwerpen (practicum Ruud lampen) AQL module

# Nieuwe ma
# Class

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 392 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            #nn.Softmax(dim=1)
        )

    # TODO Eigen train methode?

    def forward(self, x):
        x = self.base(x)
        x = self.head(x)
        return x

# my_model = ConvNet()

# print(my_model)