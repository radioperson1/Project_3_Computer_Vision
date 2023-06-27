# imports 

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

# imports 

import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

from data_pipeline import MyDataset


# Build pipeline
# TO DO: Van train validatie set halen. (datasplitting, hoeveel van de train?) Data splitting V
# TODO Eigen train methode?
# Hyperparameters checken en evt tweaken
# Confusion matrix: als ie fout is wat is er dan fout. 
# Model pickelen (storage)
# AQL ontwerpen (practicum Ruud lampen) AQL module V

# Nieuwe ma
# Class1
import torch.nn as nn
import sys
import os

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 44, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(44 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, 96)
        self.fc3 = nn.Linear(96, self.num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
    
        return x


# print(my_model)

# # Global variables

# Relative paths
# train_path = "../data/transformed_images/train"
# test_path = "../data/transformed_images/test"

# Path
train_path = "C:/Users/31641/Workspace/Project_3_Computer_Vision/data/transformed_images/train"
test_path = "C:/Users/31641/Workspace/Project_3_Computer_Vision/data/transformed_images/test"

train_transforms = transforms.Compose([
                        transforms.Resize((224,224)), 
                        transforms.ToTensor(), 
                        transforms.Normalize([0.6327, 0.5601, 0.4343], [0.2058, 0.2333, 0.2498], inplace=False) # -> default waarde, maakt een kopie en vervangt niet origineel
                        ])

# Train, test, valid dataset
train_dataset = MyDataset(train_path, train_transforms)
test_dataset = MyDataset(test_path, train_transforms)
my_imagefolder = ImageFolder(train_path, train_transforms)

# train_resized_dataset = int(len(train_dataset) * 0.8)
# valid_set = len(train_dataset) - train_resized_dataset

#valid_dataset = random_split(range(10), [3, 7], generator=generator1)

# Loaders
# batch_size_valid = len(valid_set)

train_loader16 = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader16 = DataLoader(test_dataset, batch_size=16, shuffle=False)
loader_fromfolder = DataLoader(my_imagefolder)
# train_loader32 = DataLoader(train_resized_dataset, batch_size=32, shuffle=True)
# test_loader32 = DataLoader(test_dataset, batch_size=32, shuffle=False)

# train_loader64 = DataLoader(train_resized_dataset, batch_size=64, shuffle=True)
# test_loader64 = DataLoader(test_dataset, batch_size=64, shuffle=False)

# valid_loader = DataLoader(valid_set, batch_size=batch_size_valid, shuffle=False)

# create model
model = CNNModel(4)



# defining loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer2 = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.03)

optimizer = optim.Adam(model.parameters(), lr = 0.001) 
# # Training function

def train_model(model, train_loader, criterion, optimizer, n_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    # train_loss = []
    # test_loss = []

    for epoch in range(n_epochs):
        print("Epoch number %d " % (epoch + 1) )
        
        running_loss = 0.0
        running_correct = 0.0
        total = 0.0
        
        for batch in train_loader:
            images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            total += labels.size(0)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # train_loss.append(running_loss)
            running_correct += (labels==predicted).sum().item()
            
        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00 * (running_correct/total)
        
        print(f'TRAINING SET - {running_correct} out of {total} correct. Epoch accuracy is {epoch_acc}. Epoch loss is {epoch_loss}')
        
    print(f'Final metrics are: acc is {epoch_acc}, and loss is {epoch_loss}')
    

def evaluate_model(model, test_loader):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    test_loss = 0
    predicted_correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            
            output = model(images)           
            
            predicted_value, predicted_label = torch.max(output.data, 1)
               
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output,labels)
            
            test_loss += loss.item()
            
            predicted_correct += (predicted_label == labels). sum().item()
            
            acc = predicted_correct / total * 100
            print(f'Acc of this batch is {acc}')
            
    epoch_loss = test_loss/len(test_loader)
    epoch_acc = 100.00 * predicted_correct/total
            
    print(f" - Prediction time..... predicted correct: {predicted_correct}  out of {total}. ({acc}% accuracy))")
    return f"epoch_acc: {epoch_acc}"


# optimizer = optim.SGD(model_resnet18_untrained.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

n_epochs = 20


# train_model(model, train_loader16, criterion, optimizer, n_epochs)
# evaluate_model(model,test_loader16)

saved_model_path = '../my_models/custom_model_1.pth'
path = 'C:/Users/31641/Workspace/Project_3_Computer_Vision/my_models/custom_model_1.pth'
torch.save(model.state_dict(), path)