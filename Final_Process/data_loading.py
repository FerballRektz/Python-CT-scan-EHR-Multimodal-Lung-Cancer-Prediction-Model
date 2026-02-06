import torch
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd 
import h5py



train_transforms = transforms.Compose([
    # Randomly flip the image top-bottom with a 40% probability.
    transforms.RandomVerticalFlip(p=0.4),

    # Randomly flip the image left-right with a 30% probability.
    transforms.RandomHorizontalFlip(p=0.3),

    # Randomly adjust brightness with a 30% probability and specified range.
    # The 'brightness' argument for ColorJitter is a factor range.
    # A range of [0.3, 1.2] means the new brightness is in [original * 0.3, original * 1.2].
    transforms.ColorJitter(brightness=(0.3,1.2)),

    # Randomly rotate the image by up to 40 degrees.
    transforms.RandomRotation(degrees=40),

    # There's no direct "zoom random" or "shear range" in torchvision.
    # We can approximate these using RandomAffine.
    # Shear range of 20% can be a degree range.
    # For zoom, you can use scale=(0.8, 1.2) to zoom in/out.
    # Height shift range of 20% can be approximated with translate.
    transforms.RandomAffine(degrees=0, shear=20, translate=(0, 0.2), scale=(0.8, 1.2))
])

class NLSTDataset(Dataset):
    def __init__(self,lung_image_directory,lung_table_directory, image_transform = None,show_pid = False, EHR_category = True):
        print("Loading Data....")
        lung_cancer_type_arr,lung_cancer_type_pid_arr = Loadh5(directory= lung_image_directory)
        Lung_Cancer_EHR = pd.read_csv(lung_table_directory)
        if EHR_category:
            self.data = NLST_ItemGenerator(lung_cancer_type_pid_arr,lung_cancer_type_arr,Lung_Cancer_EHR)
        print("Loading Done!")
        self.transform = image_transform
        self.show_pid = show_pid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and transform image
        pid  = item['pid']
        image = torch.tensor(item['image'],dtype=torch.float32).unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(item['label'],dtype=torch.long)
        table_data = torch.tensor(item['table'],dtype=torch.float32)
        if self.show_pid:
            return {'pid':pid,'image':image,'table':table_data}, label

        return {'image': image, 'table': table_data}, label
    
    def get_all_pids(self):
        pid_list = [item['pid'] for item in self.data]
        return pid_list


def NLST_ItemGenerator(array_pids,array_images,lungcancer_df,dict_labels={'No Lung Cancer': 0,'Stage IV':1,'Stage IA':1, 'Stage IB':1, 'Stage IIA':1, 'Stage IIIA':1,'Stage IIIB':1, 'Stage IIB':1},category = False):
    dict_list = []
    dict_label_list = dict_labels
    for index,dict_items in enumerate(array_pids):
        dataframe = lungcancer_df[lungcancer_df['ID']==dict_items]
        table_item = dataframe.iloc[:,1:-1].to_numpy()[0]
        label_item = dict_label_list[dataframe['Stage AJCC7th'].to_numpy()[0]]
        dict_list.append({'pid': dict_items, 'image':array_images[index],'table':table_item,'label':label_item})
    return dict_list


# save file algorithms
def Saveh5(directory,annotated_data,pid_data,compression_level = 5):
    with h5py.File(directory + '.h5', 'w') as hf:
        # The key is adding the compression arguments
        # compression_opts is the gzip level (1-9), 9 is max compression
        hf.create_dataset('image_data', data=annotated_data, compression='gzip', compression_opts=compression_level)
        hf.create_dataset('pid_data',data=pid_data)

def Loadh5(directory):
    with h5py.File(directory, 'r') as hf:
        # Accessing the dataset is the same as always
        retrieved_data = hf['image_data'][:] 
        pid_data = hf['pid_data'][:]
    return retrieved_data,pid_data


# dicom printers
def print_dicomNM(array,rows,columns,index_change = 0,figsize=(30,20),nii_format = False):
    slice_counter = 0
    if nii_format:
        fig, axis = plt.subplots(rows,columns, figsize = figsize)
        for i in range(rows):
            for j in range(columns):
                axis[i][j].imshow(array[:,:,slice_counter+index_change][0],cmap ='grey')
                slice_counter+=1    
        return
    else:
        if rows == 1 and columns == 1:
            fig  = plt.figure(figsize=figsize)
            plt.imshow(array)
            return
        if rows == 1 and columns > 1:
            fig, axis = plt.subplots(rows,columns, figsize = figsize)
            for j in range(columns):
                axis[j].imshow(array[slice_counter + index_change][0],cmap ='grey')
                slice_counter+=1
            return
        else:
            for i in range(rows):
                for j in range(columns):
                    axis[i][j].imshow(array[slice_counter + index_change][0],cmap ='grey')
                    slice_counter+=1
            return
        