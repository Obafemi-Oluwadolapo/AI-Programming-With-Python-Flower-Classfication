def check_command_line_arguments(args):
    """
    Classifying Images - Command Line Arguments
    Prints each of the command line arguments passed in as parameter args, 
    assumes you defined all three command line arguments as outlined in 
    'Command Line Arguments'
    Parameters:
     args -data structure that stores the command line arguments object
    Returns:
     Nothing - just prints to console  
    """
    if args is None:
        print("* Doesn't Check the Command Line Arguments because 'args_input' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:\n     dir =", args.data_dir, 
              "\n    model =", args.model_input, "\n GPU mode =", args.gpu)