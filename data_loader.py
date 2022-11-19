import numpy as np
import torch
from torchvision import datasets, transforms, models


def load_data(data_dir):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    
    return trainloader, validloader, testloader, train_datasets.class_to_idx