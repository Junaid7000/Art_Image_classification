'''
Function to train model

model used resnet:
''' 
import torch
from torchvision import models
from torch import nn, optim
import time


def train_model(model, dataloader, optimizer, loss_function, device, num_epoch):
    '''
    Function to train model
    '''
    print(f'Training model......')

    best_acc = 0.0
    model = model.to(device)
    #iterate over epoch
    for epoch in range(num_epoch):
        print(f'---------------- Epoch {epoch+1}/{num_epoch + 1}')
        #iterate over phase 
        for phase, loader in data_loader.items():

            runnig_loss = 0
            running_correct = 0
            
            #check phase of the training
            if phase == 'train':
                model.train()
            if phase == 'test':
                model.eval()

            
            #iterate over data in loader
            for data in loader:
                
        #### CHECK keys IN YOUR DATA LOADER AND CORRECT IT.
                images,labels = data['image'], data['label'] 
                #move data to device
                images, labels = images.to(device), labels.to(device)

                #set gradient of optimizer as zero
                optimizer.zero_grad()
                
                #forward pass
                outputs = model(images)
                _, preds = torch.max(outputs, 1)            #this to get predicted label

                loss = loss_function(outputs, labels)       #loss calculation

                runnig_loss+= loss.item()
                running_correct += torch.sum(preds == labels.data).item()      #corecct prediction

            epoch_loss = runnig_loss/len(loader)
            epoch_corr = running_correct/len(loader)

            #save checckpoints
            if best_acc < epoch_corr and phase == 'test':

                #save chechkpoint
                torch.save(model, 'checkpoint.pth')
                best_acc = epoch_corr
                print('checkpoint saved ......')

            print(f' {phase} : loss: {epoch_loss} - accuracy: {epoch_corr}')
    
    return model


def train_init(data_loader, num_out, num_epoch = 25, lr = 0.01):
    '''
    This function intialize training process for the given model

    model ResNet152

    num_out is number of unique labels to be predicted
    '''
    model = models.resnet152(pretrained=True)

    #change output layer of model
    num_fts = model.fc.in_features

    model.fc = nn.Linear(num_fts, num_out)

    loss_fn = nn.CrossEntropyLoss()
    optimzer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # this to keep track
    since = time.time()
    model = train_model(model, data_loader, optimzer, loss_fn, device, num_epoch)

    print(f'Total time elapsed for training : {time.time() - since}')







if __name__ == "__main__":

    random_img = torch.randn(1, 3, 224, 224)

    random_loader = [{'image': random_img, 'label': torch.tensor([0])} , {'image': random_img, 'label': torch.tensor([0])}]
    data_loader = {'train': random_loader, 'test': random_loader}

    # comment above part and include your data loader above
    train_init(data_loader, 3 )



    
