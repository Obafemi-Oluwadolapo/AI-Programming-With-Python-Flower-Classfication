import torch
import time

# Validation Model
def validation(model, testloader, criterion, device):
    validation_loss = 0
    accuracy = 0
    
    model.to(device)
    
    with torch.no_grad():
        model.eval()
        
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            
            validation_loss += criterion(model.forward(images), labels)
            
            #Probabilities of the class
            ps = torch.exp(model(images))
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            
    return validation_loss, accuracy
    
    
# train model function   
#train_losses, validation_losses = [], []
def train(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cuda'):
    
    steps = 0
    
    # Change to train mode if not already
    model.train()
    # change to cuda
    model.to(device)

    best_accuracy = 0
    #train_losses, validation_losses = [], []
    for e in range(epochs):
        # keep track of duration of each epoch
        since = time.time()
        running_loss = 0

        for (images, labels) in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format((accuracy/len(validloader))*100))
                
                # Append the values of the loss of the training loss and validation loss
                #train_losses.append(running_loss/len(trainloader))
                #validation_losses.append(validation_loss/len(testloader))
                
                model.train()
                
                running_loss = 0
                
        elapsed_time = time.time() - since
        print()
        print("Epoch completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))


