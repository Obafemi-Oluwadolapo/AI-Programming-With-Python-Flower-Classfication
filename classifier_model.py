import torch
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

def classifier_model(model_input, hidden_units, learning_rate, class_idx_mapping=None):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    #Conditional statements for different classification models
    if model_input == 'vgg13':
        model = models.vgg13(pretrained=True)

    elif model_input == 'vgg19':
        model = models.vgg19(pretrained=True)

    elif model_input == 'densenet161':
        model = models.densenet161(pretrained=True)

    elif model_input == 'densenet169':
        model = models.densenet169(pretrained=True)

    elif model_input == 'resnet101':
        model = models.resnet101(pretrained=True)

    elif model_input == 'resnet152':
        model = models.resnet152(pretrained=True)

    #Freeze parameters to avoid backpropagation
    for param in model.parameters():
        param.requires_grad=False

    #The Newly created modules have the require_grad=True by default
    if 'vgg' in model_input:
        num_features = 25088
        model.classifier = nn.Sequential(nn.Linear(num_features,hidden_units),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(hidden_units,512),
                                            nn.ReLU(),
                                            nn.Linear(512,len(cat_to_name)),
                                            nn.LogSoftmax(dim=1))
        model.classifier
        
    elif 'resnet' in model_input:
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_features, hidden_units),
                                nn.ReLU(),
                                nn.BatchNorm1d(hidden_units),
                                nn.Dropout(p=0.5),
                                nn.Linear(hidden_units,len(cat_to_name)),
                                nn.LogSoftmax(dim=1))
        model.fc
        
    elif 'densenet' in model_input:
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Linear(num_features,hidden_units),
                                            nn.ReLU(),
                                            nn.Dropout(p=0.5),
                                            nn.Linear(hidden_units,512),
                                            nn.ReLU(),
                                            nn.Linear(512,len(cat_to_name)),
                                            nn.LogSoftmax(dim=1))
        model.classifier
    #print(model.__class__.__name__)
    
    # Loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
    model.class_idx_mapping = class_idx_mapping
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return model, criterion, optimizer