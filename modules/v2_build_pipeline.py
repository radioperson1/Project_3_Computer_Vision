import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pickle

# ------------------------------------------------------------------------
from data_pipeline import MyDataSet

# Global variables

train_path = "C:/Users/31641/Workspace/Project_3_Computer_Vision/data/transformed_images/train"
test_path = "C:/Users/31641/Workspace/Project_3_Computer_Vision/data/transformed_images/test"

# Train, test, valid dataset
train_dataset = MyDataSet(train_path)
test_dataset = MyDataSet(test_path)

train_resized_dataset = int(len(train_dataset) * 0.8) 
# valid_set = len(train_dataset) - train_resized_dataset ## > niet goed ga naar images

# Loaders

# batch_size_valid = len(valid_set)

# train_loader = DataLoader(train_resized_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# valid_loader = DataLoader(valid_set, batch_size=batch_size_valid, shuffle=False)

# -----------------------------------------------------------------------------------


# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
                
#         self.base = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 392 * 2, 128),
#             nn.ReLU(),
#             nn.Linear(128, 4),
#             nn.Softmax(dim=1)
#         )

#     # TODO Eigen train methode?

#     def forward(self, x):
#         x = self.base(x)
#         x = self.head(x)
#         return x


# class MyModel(ConvNet):
#     def __init__(self):
#         super().__init__()
#         # Define your model architecture and other components here
#         self.model = ConvNet()
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.03)
#         self.train_losses = []
#         self.test_losses = []
        
#     def training(self, num_epochs, train_loader, test_loader):
        
#         self.model.train() # --> set to train mode
         
#         # Training loop       
        
#         for n in range(num_epochs):
            
            
#             train_loss = 0.0
#             train_correct = 0.0
#             total_predicted = 0.0
            
            
#             for batch in train_loader:
#                 # Forward pass
#                 images, labels = batch
#                 images = images
#                 labels = labels
#                 total_predicted += labels.size(0)
            
#                 #images = images.to(device)
#                 #labels = labels.to(device)
                
#                 self.optimizer.zero_grad()
#                 outputs = self.model(images)
                
#                 values, predicted_labels = torch.max(outputs.data, 1)
#                 loss = self.loss_fn(outputs, labels)

#                 # Backward and optimize

#                 loss.backward()
#                 self.optimizer.step()
                
#                 # 
#                 train_loss += loss.item()
#                 train_correct += (labels==predicted_labels).sum().item()

#             self.train_losses.append(train_loss / len(train_loader))

#             test_loss = self.evaluate(test_loader)
#             self.test_losses.append(test_loss)
            
#         epoch_acc = round(100.0 * train_correct / total_predicted, 2)
#         epoch_loss = round(train_loss / len(train_loader), 2)
        
#         print(f'Final metrics of trained model: accuracy is {epoch_acc}, loss is {epoch_loss}')
        
#         # Print training progress
            
#     def evaluate(self, test_loader):
#         # Evaluation loop
#         self.model.eval()
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for images, labels in test_loader:
#                 outputs = self.model(images)
#                 _, predicted = torch.max(outputs.images, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         accuracy = (correct / total) * 100
#         print(f'Accuracy of the model after evaluation is  {accuracy} %')
            
        
#     def plot_losses(self):
#         plt.plot(self.train_losses, label='Train Loss')
#         plt.plot(self.test_losses, label='Test Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.show()

#     def save_model(self, filepath):
#         # Save model to a dictionary
#         model_state = self.model.state_dict()
#         torch.save(model_state, filepath)

#     def load_model(self, filepath):
#         # Load model from a dictionary
#         model_state = torch.load(filepath)
#         self.model.load_state_dict(model_state)



# # Usage example
# model = MyModel()

# # Training
# model.train(15, train_loader, test_loader)

# # Evaluation
# model.evaluate(test_loader)

# # Save model
# filepath = 'C:/Users/31641/Workspace/Project_3_Computer_Vision/models'
# model.save_model('model.pkl')

# # Load model
# filepath = 'C:/Users/31641/Workspace/Project_3_Computer_Vision/models'
# model.load_model('model.pkl')
