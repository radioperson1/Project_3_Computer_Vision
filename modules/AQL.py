import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from collections import Counter

from network import CNNModel


model = CNNModel(4)
path = 'C:/Users/31641/Workspace/Project_3_Computer_Vision/my_models/custom_model_1.pth'
model_state_dict = torch.load(path)
model.load_state_dict(model_state_dict)
model.eval()

mapping = {'Blotch_Apple': 0, 'Normal_Apple': 1, 'Rot_Apple': 2, 'Scab_Apple': 3}

reversed_mapping = {v: k for k,v in mapping.items()}

#Alle paden naar de foto's in een lijst zetten

pil_images = []
batch_folder_path = 'C:/Users/31641/Workspace/Project_3_Computer_Vision/data/batch'

for filename in os.listdir(batch_folder_path):
    file_path = os.path.join(batch_folder_path, filename)
    pil_images.append(Image.open(file_path))



transform = transforms.Compose([
                        transforms.Resize((224,224)), 
                        transforms.ToTensor(), 
                        transforms.Normalize([0.6327, 0.5601, 0.4343], [0.2058, 0.2333, 0.2498], inplace=False) # -> default waarde, maakt een kopie en vervangt niet origineel
                        ]) 

tensor_list = [transform(img) for img in pil_images]

tensor_stack = torch.stack(tensor_list)

output = model(tensor_stack)

_, predicted = torch.max(output.data, 1)

predicted_list = predicted.tolist()
print(predicted_list)

predicted_labels = [reversed_mapping[v] for v in predicted_list]
print(predicted_labels)

predicted_dictionary = dict(Counter(predicted_labels))
print(predicted_dictionary)

AQL_limiet = 7
batch_size = len(pil_images)

if predicted_dictionary["Normal_Apple"] < (batch_size - AQL_limiet):
    print('Batch rejected')

else:
    print('Batch approved')


'''
tensor([[ 0.2562, -3.4506,  4.6048, -0.5037],
        [-0.5697,  3.0842, -1.1880, -1.7482],
        [-0.4854,  3.3350, -0.4359, -2.7005],
        [-1.6833,  4.0472, -1.2948, -1.3231],
        [ 1.0089, -1.0196, -2.0450,  1.5581],
        [-1.8442,  0.6840, -0.4006,  2.2394],
        [-1.3261,  5.4487, -1.1063, -3.0197],
        [-1.7893,  2.3551, -1.5360,  0.5417],
        [-1.3411,  5.0628, -1.0799, -2.7068],
        [-1.5935, -0.8936,  3.6085, -0.4299],
        [-2.0276,  6.4815, -0.5812, -3.8004],
        [-2.6180,  4.5136, -0.3584, -1.0848],
        [-1.5814,  3.4025, -0.7011, -1.2118],
        [-2.4450,  5.4681, -2.2027, -0.5819],
        [-1.4689,  5.8526, -1.3628, -3.0047],
        [-2.0607,  1.0866, -2.0302,  3.3750],
        [ 0.1439, -8.2053,  8.1141,  2.8545],
        [-7.1797, -6.6877,  5.4569, 16.7887],
        [ 8.8821, -6.7135, -4.8156,  1.4853],
        [-1.2377,  1.8724, -4.6225,  3.9035]], grad_fn=<AddmmBackward0>)

tensor([2, 1, 1, 1, 3, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 2, 3, 0, 3])

'''