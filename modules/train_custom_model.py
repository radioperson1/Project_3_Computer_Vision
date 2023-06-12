import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from custom_model import ConvNet
from training_data_pipeline import DataClass



def train_model(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    #device = set_device()
    best_acc = 0
    train_loss = []
    test_loss = []
    
    loss = nn.CrossEntropyLoss()


    for epoch in range(n_epochs):
        print("Epoch number %d " % (epoch + 1) )
        
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        
        for batch in train_loader:
            images, labels = batch
            images = images
            labels = labels
            total += labels.size(0)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
    
            l = loss(outputs, labels)
            
            l.backward()
            
            optimizer.step()
            
            running_loss += l.item()
            train_loss.append(running_loss)
            
            running_correct += (labels==predicted).sum().item()
            
        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00 * running_correct/total
        
        print(" - Training dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f"
              % (running_correct, total, epoch_acc, epoch_loss))
    
    test_total = 0
    test_correct_fn = 0
    test_loss_fn = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images
            labels = labels 
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            l = loss(outputs, labels)
            
            test_loss_fn += l.item()
            test_loss.append(test_loss_fn)
            
            test_total += labels.size()[0]
            
            test_correct_fn += (predicted == labels).sum().item()
            
            acc = 100.00 * test_correct_fn/test_total
    
    print(f" - Testing dataset. Got {test_correct_fn} out of {test_total} images correctly. Accuracy: {acc}%")
    
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



model = ConvNet()

criterion = nn.CrossEntropyLoss()  # experiment with different loss functions

# optimizer = optim.SGD(model_resnet18_untrained.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)
optimizer = optim.Adam(model.parameters())
n_epochs = 15


##### INPUT

train_path = "C:/Users/31641/Workspace/Project_3_Computer_Vision/data/images/train"
test_path = "C:/Users/31641/Workspace/Project_3_Computer_Vision/data/images/test"

data_pipeline = DataClass(train_path, test_path)

train_loader = data_pipeline.train_loader
test_loader = data_pipeline.test_loader

# train_model(model, train_loader, test_loader, criterion, optimizer, n_epochs)

print(type(train_loader))