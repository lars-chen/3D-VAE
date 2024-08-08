import numpy as np
import pandas as pd
import torch
import nibabel as nib
from  scipy import ndimage
from torch.utils.data.dataset import Dataset

## 3D Brain Imaging Dataset was taken from Nvidia 100k brain imaging data
# https://blogs.nvidia.com/blog/ai-brain-images-kcl/

def resize_data_volume_by_scale(data, scale):
   """
   Resize the data based on the provided scale
   """
   scale_list = [scale,scale,scale]
   return ndimage.interpolation.zoom(data, scale_list, order=0)

class  NvidiaDataset(Dataset):
    def __init__(self, data_dir = "../atrophy_bet"):
        super().__init__()
        
        
        self.data_dir = data_dir
        self.subjects = pd.read_csv(self.data_dir + "/participants.csv")
        
        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, index):
        # load labels
        item = dict()
        item["age"] = self.subjects["age"][index]
        item["sex"] = 1 if self.subjects["sex"][index] == "M" else 0
        item["brain_volume"] = (self.subjects["brain_vol"][index] - 1100000)/(1800000 - 1100000) # min max normalize
        item["ventricle_volume"] = (self.subjects["ventricle_vol"][index] - 7000)/(170000 - 7000) # min max normalize
    
        # load image
        participant_id = str(self.subjects["subject"][index])
        participant_id = '00000'[:5-len(participant_id)] + participant_id
        img_dir = f"{self.data_dir}/{participant_id}.nii.gz"
        img = nib.load(img_dir).get_fdata()[12:148, 8:212, :136] # crop excess black pixels
        img = resize_data_volume_by_scale(img, 0.94)[np.newaxis, :, :, :]
        item["image"] = img    
        return item
    
    @staticmethod
    def _prepare_item(item):
        eps = 1e-10
        item["age"] = torch.as_tensor(item["age"], dtype=torch.float) 
        item["sex"] = torch.as_tensor(item["sex"], dtype=torch.float)
        item["brain_volume"] = torch.as_tensor(item["brain_volume"], dtype=torch.float) + eps
        item["ventricle_volume"] = torch.as_tensor(item["ventricle_volume"], dtype=torch.float) + eps
        item["image"] = torch.as_tensor(item["image"][10:150, 20:202, :140], dtype=torch.float)
        return item
    
    
