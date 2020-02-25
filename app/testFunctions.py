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

    class_to_idx = {'': 0



    }

    return dataloaders_dict

def test_model(model, dataloaders, device):
    since = time.time()

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, _ in dataloaders['test']:
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.max(outputs, 1)[1].cpu().data
            preds = preds.numpy().squeeze()
            print(preds)
            
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return preds
