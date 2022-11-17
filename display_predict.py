from predict_flower import predict

def display_predict(image_path, model_file):
    """
    Display th Image  and preditions from the model
    
    Prediction of the Images where:
    probs = Probability of Images between 0 and 1 
    classes = the flowers the images might be
    """
    class_idx_mapping = train_datasets.class_to_idx
    idx_class_mapping = {v: k for k, v in class_idx_mapping.items()} 
    
    # Prediction of the image
    probs, classes = predict(image_path, model_file, idx_class_mapping=idx_class_mapping)
    
    class_names = [cat_to_name[c] for c in classes]
    
    # Plots of the flower and Probability
    fig, (ax1, ax2) = plt.subplots(nrows=2,figsize=(4,10))
    class_number = image_path.split("/")[-2]
    title = cat_to_name[str(class_number)]
    imshow(process_image(image_path), ax1, title)
    print(title)

    pbs = (probs[0][0]) * 1
    #print(pbs)
    scalars = [*range(len(probs[0][0]))]
    #print(scalars)
    ax2.barh(scalars, pbs)
    plt.xlabel("Probability")
    plt.yticks(scalars, class_names)
    plt.show()