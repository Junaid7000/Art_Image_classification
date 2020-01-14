from data_to_csv import get_data_to_csv
from ArtImageLoader import ArtImageData
from torch.utils.data import DataLoader
from pipeline import train
from torchvision import models,transforms
import pandas as pd
import torch
from torch import nn
import torch.optim as optim

if __name__ == "__main__":
    #Constants
    root='E:/Developer/Machine Learning and Deep Learning/Pytorch_Training/Art_Image_classification/dataset/dataset_updated'
    #creat image_label_csv
    #get_data_to_csv(root)

    import os
    train_loc = root + '/training_set.csv'
    val_loc = root + '/training_set.csv'

    train_csv=pd.read_csv(root+'/training_set.csv')
    test_csv=pd.read_csv(root+'/validation_set.csv')
    num_epochs=25
    batch_size=7
    num_workers=0
    transform= transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
    running_loss=0.0
    train_loss=0.0

    #train_test_set
    train_set=ArtImageData(train_loc,transform=transform)
    test_set=ArtImageData(val_loc,transform=transform)

    #train_test_loader
    train_loader=DataLoader(train_set, batch_size=batch_size)
    test_loader=DataLoader(test_set, batch_size=batch_size)

    #using Resnet18 and downloading pretrained weights
    model=models.resnet152(pretrained=True)
    num_ft = model.fc.in_features
    model.fc = nn.Linear(num_ft, 5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    #test_model
    #t1 = torch.randn(1, 3, 224, 224)
    #out = model(t1)
    #print(out.shape)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
                
    #loader_key_value_pair
    loader_dict={'trainloader_key': train_loader, 'testloader_key': test_loader}
    best_loss = 10
    
    #training
    for epochs in range (num_epochs):
        print("-----------{}--------".format(epochs))
        train_loss = 0
        test_loss = 0
        
        for key,value in loader_dict.items():
            #value==train_loader
            if key == 'trainloader_key':
                #Loss Function and Optimizer
                optimizer.zero_grad()
                for data in value:
                    images,labels=data['image'], data['label']

                    model,images,labels=model.to(device),images.to(device),labels.to(device)

                    outputs=model(images)
                    loss=criterion(outputs,labels)
                    train_loss+=loss.item()

                train_loss = train_loss/len(train_loader)
                print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                    epochs+1, 
                    train_loss
                    ))
            else:
            #value==test_loader
                model.eval()
                for data in value:
                    images,labels=data['image'], data['label']
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                    loss = criterion(outputs, labels)
                    test_loss +=loss.item()

                test_loss = test_loss/len(test_loader)
                print('Epoch: {} \t Testing Loss: {:.6f}'.format(
                    epochs+1, 
                    test_loss
                    ))

                if test_loss < best_loss:
                    best_loss = test_loss
                    print(f'saving checkpoint at epoch {epochs}')
                    torch.save(model, './checkpoint.pt')

            

                


        




                    







    


