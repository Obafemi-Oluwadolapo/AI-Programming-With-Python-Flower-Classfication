import argparse

def args_input():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    #### Creates Argument Parser object named parser
    parser = argparse.ArgumentParser(description = 'Process some images.')

    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    # Argument 1: Path to a folder
    parser.add_argument('--data_dir', type=str, default='flowers/',
                        help = 'path to the folder of train, validation and test datasets subfolder') 
    parser.add_argument('--print_every', type = int, default = 20, 
                        help = 'Number sub epochs of training for each testing pass (for each epoch): default=1)')
    parser.add_argument('--model_file', type=str, default='', 
                        help = 'directory to the file to save checkpoints: default="" {root folder}')
    parser.add_argument('--model_input', type=str, default='vgg13',
                        help='CNN Model Architecture: default = vgg13')
    ##parser.add_argument('--model', type=str, default='vgg13',
                        #help='CNN Model Architecture: default = vgg13')
    parser.add_argument('--learning_rate', type=int, default=0.01,
                        help='Learning rate: default=0.0002')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of Epochs for training: default=20')
    parser.add_argument('--hidden_units', type=int, default=4096,
                        help='Number of hidden layer units needed in the NN model for training: default=512')
    parser.add_argument('--gpu', default=True, action='store_true',
                        help='Define usage of GPU Cuda device: default=True')
    
    return parser.parse_args()
                        
                 
def args_input_predict():
    """ Process input arguments for the prediction script
    """
                        
    # Create Parse using ArgumentParser
    #### Creates Argument Parser object named parser
    parser = argparse.ArgumentParser(description = 'Process some images.')                 
    
    parser.add_argument('--image_path', type=str, default='flowers/valid/29/image_04108.jpg',
                        help = 'path to image directory for prediction')
    parser.add_argument('--model_file', type=str, default='',
                        help = 'path to checkpoint directory for saving model prediction')  
    
    # Optional Arguments
    parser.add_argument('--gpu', default=True, action='store_true',
                        help='Define usage of GPU Cuda device: default=True')
    parser.add_argument('--topk', type=int, default=5,
                        help='Define usage of GPU Cuda device: default=True')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help = 'path to checkpoint directory for saving model prediction')
    
    return parser.parse_args()