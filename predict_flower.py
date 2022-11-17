# Imports Function from their different .py files
from args_inputs import args_input, args_input_predict
from load_checkpoint import load_checkpoint
from process_image import process_image
from data_loader import load_data

args_predict = args_input_predict()
args = args_input()

def predict(image_path, model_file, topk, device, idx_class_mapping):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Arguments:
        image_path: Path to the image
        model: Trained model
    Returns:
        classes: Top k class numbers.
        probs: Probabilities corresponding to those classes
    '''
    
    _, _, idx_class_mapping = load_data(args.data_dir)
    # Build the model from the checkpoint
    model, optimizer = load_checkpoint(model_file)
    
    # No need for GPU
    model.to(device)
    
    model.eval()
     
    img = process_image(image_path)
    img = np.expand_dims(img, axis=0)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    
    with torch.no_grad():
        log_probabilities = model.forward(img_tensor)
    
    probabilities = torch.exp(log_probabilities)
    probs, indices = probabilities.topk(topk)
    
    probs = [[probs.numpy().squeeze()]]
    indices = indices.numpy().squeeze()
    classes = [idx_class_mapping[index] for index in indices]
    
    return probs, classes