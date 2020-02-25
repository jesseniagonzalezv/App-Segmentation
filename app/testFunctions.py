import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import argparse
from torchvision import datasets, models, transforms
import time
import os

def dataTranforms(input_size):
    # Data augmentation and normalization for training
    # Todo (improvements) --> this should be in a dataloader class to avoid using augmented data in validation 
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])   
    }
    return data_transforms

def dataLoaders(input_size, data_dir):
    # Same process as training
    data_transforms = dataTranforms(input_size)
    image_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}
    dataloaders_dict = {'test': torch.utils.data.DataLoader(image_dataset['test'], batch_size=1, shuffle=True, num_workers=4)}

    class_to_idx = {'aceite': 0, 
                    'agua': 1, 
                    'arroz': 2, 
                    'azucar': 3, 
                    'cafe': 4, 
                    'caramelo': 5, 
                    'cereal': 6, 
                    'chips': 7, 
                    'chocolate': 8, 
                    'especias': 9, 
                    'frijoles': 10, 
                    'gaseosa': 11, 
                    'harina': 12, 
                    'jugo': 13, 
                    'leche': 14, 
                    'maiz': 15, 
                    'mermelada': 16, 
                    'miel': 17, 
                    'nueces': 18, 
                    'pasta': 19, 
                    'pescado': 20, 
                    'salsatomate': 21, 
                    'te': 22, 
                    'torta': 23, 
                    'vinagre': 24}

    return dataloaders_dict, class_to_idx, image_dataset

def test_model(model, dataloaders, device, class_to_idx):
    since = time.time()
    class_name = []

    # Iterate over data.
    for inputs, _ in dataloaders['test']:
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.max(outputs, 1)[1].cpu().data
            preds = preds.numpy().squeeze()
            prediction = list(class_to_idx.keys())[list(class_to_idx.values()).index(preds)]
            class_name.append(prediction)
            
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return class_name
