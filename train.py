# Imports here
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os
from time import time
import copy
import json
from PIL import Image


# Import from other Functions from different files
from print_functions_for_lab_checks import *

from args_inputs import args_input
from data_loader import load_data
from save_checkpoint import save_checkpoint
from check_accuracy import check_accuracy_test
from classifier_model import classifier_model
from train_model import train

        
        
def main():
    # Calls all the variables in the args_input function
    args = args_input()
    
    #os.system("mkdir -p " + args.model_file)
    
    # Function that checks command line arguments using in_arg  
    check_command_line_arguments(args)
    
    print('------------------------------------------')
    print('All ready! Now preparing to train model...')
    print('...')
    print()
    
    print('Loading data and pre-processing...')
    print('...')
    print()
    
    # Load data datasets from load_data function
    trainloader, validloader, class_idx_mapping = load_data(data_dir=args.data_dir)
    
    # Opening the json file for the names of the flowers
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    print()
    print('Preparing to train model!')
    print('Building network...')
    print('...')
    print()
    
    # print the classifier model, criterion and optimizer
    model, criterion, optimizer = classifier_model(model_input=args.model_input,
                                                     hidden_units=args.hidden_units,
                                                     learning_rate=args.learning_rate,
                                                     class_idx_mapping=class_idx_mapping)
    
    print(model)
    print('Generated Model has been printed!')
    print()
    print()
    
    # check if the GPU mode is True
    device = None
    if args.gpu:
        device = "cuda"
    else:
        device = "cpu"
    
    print('Commence training...')
    print('...')
    print()
    
    train(model, 
        trainloader=trainloader, 
        validloader=validloader,
        epochs=args.epochs, 
        print_every=args.print_every, 
        criterion=criterion,
        optimizer=optimizer,
        device="cuda")
    
    _,_,class_t_idx = load_data(args.data_dir)
    print('Saving the best model...')
    save_checkpoint(model, 
                'model_flower_classifier_vgg13.pt',
                class_t_idx)

    
    print('Training Complete!')
    print()
    print()
    
    # Checking for accuracy on the test dataset
    print('Checking for accuracy on the test datasets....')
    test_accuracy = check_accuracy_test(testloader, model)
    print('Accuracy of the network on the 10000 test images: %d %%' % test_accuracy)    

if __name__ == '__main__':
    main()