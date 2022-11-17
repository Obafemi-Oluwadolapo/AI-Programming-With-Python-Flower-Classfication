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
import time
import copy
import json
from PIL import Image

# Imports Function from their different .py files
from args_inputs import args_input, args_input_predict
from data_loader import load_data
from load_checkpoint import load_checkpoint
from process_image import process_image
from imshow import imshow
from predict_flower import predict
from display_predict  import display_predict


def main():
    args = args_input()
    args_predict = args_input_predict()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args_predict.gpu and device == "cuda":
        device = "cuda"
    elif args_predict.gpu and device == "cpu":
        print("CUDA not found on device, using CPU instead!")
        device = "cpu"
    else:
        device = "cpu"
    
    # DEfining the idx_class_mapping from load_data function
    _, _, idx_class_mapping = load_data(args.data_dir)
    
    probs, classes = predict(image_path=args_predict.image_path, 
                             model_file=args_predict.model_file, 
                             topk=args_predict.topk, 
                             device="cpu",
                             idx_class_mapping=idx_class_mapping)
    
    print('Commencing Prediction...')
    print('...')
    print()
    
    print("Predicting what kind of flower this is...")
    print("...")
    print("...")
    print('This flower is most likely a {} with a {:.6f} probability!'.format(classes[0], probs[0][0]))
    # Opening the json file for the names of the flowers
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    class_names = [cat_to_name[c] for c in classes]
    
    x = PrettyTable()
    x.field_names = ["Class Name", "Probability"]
    for c,p in zip(class_names, probs[0][0]):
        x.add_row([c, p])
    
    print(x)
    
if __name__ == '__main__':
    main()
    
    
    