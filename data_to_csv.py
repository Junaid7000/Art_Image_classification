''''
This function is used to get data's location and label from in a csv file 
'''

import pandas as pd
import numpy as np
import os
import time
import cv2

def get_data_to_csv(root_loc):
    ''''
    This function convert given data location folder to a csv file.

    root_loc: root location of dataset folder
    '''

    # get list of sub dir
    list_dir = os.listdir(root_loc)

    for sub_dir in list_dir:
        
        img_loc_list = []
        labels_list = []
        # get location of sub dir
        sub_dir = os.path.join(root_loc, sub_dir)

        for labels in os.listdir(sub_dir):
            labels_loc = os.path.join(sub_dir, labels)
            #iterate over images
            for images in os.listdir(labels_loc):

                try:
                    img_loc = os.path.join(labels_loc, images)
                    img = cv2.imread(img_loc).shape
                    img_loc_list.append(img_loc)
                    labels_list.append(labels)

                except:
                    print('img unavialble')

        data_dict = {'img_loc': img_loc_list, 'labels': labels_list}
        df = pd.DataFrame(data_dict)    
        save_loc = os.path.join(root_loc, sub_dir + '.csv' )
        df.to_csv(save_loc)
        print('csv saved')

        
                
                










if __name__ == "__main__":

    #root_loc = 'D:/Art data/dataset/dataset_updated'
    root_loc = 'E:/Developer/Machine Learning and Deep Learning/Pytorch_Training/Art_Image_classification/dataset/dataset_updated'

    since = time.time()
    get_data_to_csv(root_loc)

    print(f'Total time elapsed:  {time.time() - since}')
