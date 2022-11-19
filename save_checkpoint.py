from args_inputs import args_input
from data_loader import load_data
from classifier_model import classifier_model
import torch

args = args_input()
trainloader, validloader, testloader, class_idx_mapping = load_data(data_dir=args.data_dir)
model, criterion, optimizer = classifier_model(model_input=args.model_input,
                                                     hidden_units=args.hidden_units,
                                                     learning_rate=args.learning_rate,
                                                     class_idx_mapping=class_idx_mapping)
    
def save_checkpoint(model,model_file, class_to_idx):
    model.class_to_idx = class_to_idx
    if 'VGG' in model.__class__.__name__:
        parameters = {
            'class_to_idx': model.class_to_idx,
            'classifier': model.classifier,
            'state_dict': model.state_dict(),
            'optimizer': optimizer,
            'optimizer_state_dict': optimizer.state_dict()
        }
        
    elif 'ResNet' in model.__class__.__name__:
        parameters = {
            'class_to_idx': model.class_to_idx,
            'classifier': model.fc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer,
            'optimizer_state_dict': optimizer.state_dict()
        }
        
    elif 'DenseNet' in model.__class__.__name__:
        parameters = {
            'class_to_idx': model.class_to_idx,
            'classifier': model.classifier,
            'state_dict': model.state_dict(),
            'optimizer': optimizer,
            'optimizer_state_dict': optimizer.state_dict()
        }
    
    torch.save(parameters, model_file)