# [START] Configuration and imports!
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import argparse
from torchvision import datasets, models, transforms
from trainFunctions import initialize_model, train_model

import matplotlib.pyplot as pyplot
import time
import os
import copy
from PIL import Image
print("Pytorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
# [END] configuration and imports!

# [START] Define data transformation for different networks
def dataTranforms(input_size):
    # Data augmentation and normalization for training
    # Todo (improvements) --> this should be in a dataloader class to avoid using augmented data in validation 
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])   
    }
    return data_transforms
# [END] data transformation function

# [START] Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not 'cuda':
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
# [END] check GPU

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--datadir', type=str, default='./dataset')
    arg('--batchSize', type=int, default=16)
    arg('--classes', type=int, default=25, help='number of classes in dataset')
    arg('--nepochs', type=int, default=50)
    arg('--model', type=str, default='resnet50', choices=['resnet50', 'densenet201', 'inception'])

    args = parser.parse_args()

    # [START] Network parameter configuration for training
    model_name = args.model # name of the model
    data_dir = args.datadir # dir of training images
    num_classes = args.classes # classes in the dataset
    batch_size = args.batchSize # batch size for training
    num_epochs = args.nepochs # number of epochs to train
    # [END] network parameter configuration

    if model_name == 'resnet50':
        # [START] Define model for training --> RESNET 50
        model_resnet50 , input_size, params_to_update = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True)
        model_resnet50 = model_resnet50.to(device)

        # Read datasets and separate them in train and validation
        data_transforms = dataTranforms(input_size)
        image_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train']}
        train_dataset, val_dataset = torch.utils.data.random_split(image_dataset['train'], [int(len(image_dataset['train'])*0.8), (len(image_dataset['train'])-int(len(image_dataset['train'])*0.8))])
        dataloaders_dict = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
                            'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)}

        optimizer_ft = optim.SGD(params_to_update, lr=0.003, momentum=0.9) # observe that all parameters are being optimized
        scheduler= torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1) # every 5 steps reduce in 0.1
        criterion = nn.CrossEntropyLoss() # setup the loss function
        model_resnet, hist = train_model(model_resnet50, dataloaders_dict, device, criterion,scheduler, optimizer_ft, num_epochs, is_inception=(model_name=="inception"))

        # Saving model and optimizer
        # torch.save(model_resnet50.state_dict(),'./models/modelResnet50.pth') 
        torch.save(model_resnet50,'./models/modelResnet50.pth') 
        # [END] defining model

    elif model_name == 'densenet201':
        # [START] Define model for training --> DENSENET 201
        model_densenet, input_size,params_to_update = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True)
        model_densenet = model_densenet.to(device)

        data_transforms = dataTranforms(input_size)
        image_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train']}
        train_dataset, val_dataset = torch.utils.data.random_split(image_dataset['train'], [int(len(image_dataset['train'])*0.8), (len(image_dataset['train'])-int(len(image_dataset['train'])*0.8))])
        dataloaders_dict = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
                            'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)}

        optimizer_ft = optim.Adagrad(model_densenet.parameters(), lr=0.003)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) # every 7 steps reduce in 0.1
        criterion = nn.CrossEntropyLoss()
        model_densenet, hist = train_model(model_densenet, dataloaders_dict, device, criterion,scheduler, optimizer_ft, num_epochs, is_inception=(model_name=="inception"))

        # Saving model and optimizer
        # torch.save(model_densenet.state_dict(),'./models/modelDensenet201.pth')
        torch.save(model_densenet,'./models/modelDensenet201.pth')
        # [END] defining model

    elif model_name == 'inception':
        # [START] Define model for training --> INCEPTION
        model_inception, input_size, params_to_update = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True)
        model_inception= model_inception.to(device)

        data_transforms = dataTranforms(input_size)
        image_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train']}
        train_dataset, val_dataset = torch.utils.data.random_split(image_dataset['train'], [int(len(image_dataset['train'])*0.8), (len(image_dataset['train'])-int(len(image_dataset['train'])*0.8))])
        dataloaders_dict = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
                            'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)}

        optimizer_ft = optim.SGD(model_inception.parameters(), lr=0.003, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1) # every 5 steps reduce in 0.1
        criterion = nn.CrossEntropyLoss()
        model_inception, hist_inception = train_model(model_inception, dataloaders_dict, device, criterion ,scheduler, optimizer_ft, num_epochs, is_inception=(model_name=="inception"))

        # Saving model and optimizer
        # torch.save(model_inception.state_dict(),'./models/modelInception.pth')
        torch.save(model_inception,'./models/modelInception.pth')
        # [END] defining model

    else:
        print('Model not defined! Exiting ...')
        exit()


if __name__ == '__main__':
    main()