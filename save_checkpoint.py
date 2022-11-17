# TODO: Save the checkpoint
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