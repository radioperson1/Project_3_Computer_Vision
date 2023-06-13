
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn

# Data pipeline
# TO DO: add normalization in transform
# TO DO: add data augmentation before Conv2d layer, je kan de transformed data toevoegen aan dataset. Wat ga ik doen met de transformede plaatjes? 2 datasets maken met transform en dan concatenaten.
# org_dataset, aug_dataset
# Nieuwe map plaatjes in een andere map saven met data_transformed (storage)

class DataSet:
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        
        # Resize    
        self.resize = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]) 

        if transform is None:
            transforms.ColorJitter()
            
            
    def prepare_data(self, data_path):    
        return ImageFolder(root=data_path, transform=self.resize)

class MyDataLoader(DataLoader):
 
    def __init__(self, data_path, batch_size=16, shuffle=True):
        super.__init__(DataSet(data_path), batch_size, shuffle)
        # self.dataset = dataset
        # self.batch_size = batch_size
        # self.shuffle = shuffle
        # # self.dataloader = None
    
    # def create_dataloader(self):
    #     self.dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle=self.shuffle)
    #     return self.dataloader
    
    # def slice_the_cake(self):
    #     for batch in self:
    #         print(batch)
        
        
        

#train, test = Datasets.run_pipeline()

# Build pipeline, confusion matrix, AQL chart
