import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np

class MyDataset(Dataset):
    def __init__(self, images_file, meta_file, label_file):
        with open(images_file, 'rb') as f:
            self.images = pickle.load(f)
        with open(meta_file, 'rb') as f:
            self.meta = pickle.load(f)
        with open(label_file, 'rb') as f:
            self.labels = pickle.load(f)
        

        assert len(self.images) == len(self.meta) == len(self.labels), "Len Error!"
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        meta = self.meta[idx]
        label = self.labels[idx]
        
        meta = np.array(meta).astype(int)
        
        image = torch.from_numpy(image)
        meta = torch.from_numpy(meta)
        label = torch.tensor(label)
        return (image, meta), label

images_file = "dataset/images.pkl"
meta_file = "dataset/meta.pickle"
label_file = "dataset/label.pickle"


dataset = MyDataset(images_file, meta_file, label_file)


batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

