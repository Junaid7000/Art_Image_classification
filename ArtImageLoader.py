import pandas as pd
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class ArtImageData(Dataset):
    #input csv_path and transform
    def __init__(self,csv_dir,transform=None):
        self.data_csv=pd.read_csv(csv_dir)
        self.label_dict={'drawings':0 , 'engraving':1 , 'icongraphy':2 , 'painting':3 , 'sculpture':4 }
        self.transform=transform

    def __len__(self):
        return len(self.data_csv)
    
    def __getitem__(self,idx):
        img=cv2.imread(self.data_csv['img_loc'][idx])
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=Image.fromarray(img)
        label=self.data_csv['labels'][idx]
        label=self.label_dict[label]

        sample={'image': img, 'label':label}

        if self.transform:
            sample['image']=self.transform(sample['image'])
        return sample






if __name__ == "__main__":
    root='E:/Developer/Machine Learning and Deep Learning/Pytorch_Training/Art_Image_classification/dataset/dataset_updated'
    train_csv=root+'/training_set.csv'
    transform=transforms.ToTensor()
    trainset=ArtImageData(train_csv,transform)
    
    sample=trainset[1]
    plt.imshow(sample['image'].numpy().transpose(1,2,0))
    plt.show()
    print(sample['label'])

    