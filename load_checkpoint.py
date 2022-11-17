import torch

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(model_file):
    checkpoint = torch.load(model_file)
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

    if 'vgg' in model_file:
        model.classifier = checkpoint['classifier']
        
    elif 'resnet' in model_file:
        model.fc = checkpoint['classifier']
    elif 'densenet' in model_input:
        model.classifier = checkpoint['classifier']
    
    # Model Specifics
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Updated optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer