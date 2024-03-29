import os 
import cv2 
import numpy as np

import torch
from torch.utils.data import Dataset

class slcset(Dataset):
    def __init__(self, data_dir, transform=None, nolabel = False):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        ## add no label 
        self.class_to_idx["#nolabel"] = -1
        self.idx_to_class[-1] = "#nolabel"
        self.nolabel = nolabel

        if(self.nolabel):
            self.class_to_idx = {}
            self.idx_to_class = {}
            self.classes = []

        self.img_paths = self.get_img_paths()
        self.shuffle()

    def get_img_paths(self):
        if(self.nolabel):
            return self.get_img_paths_nolabel()
        img_paths = []
        for cls in self.classes:
            cls_dir = os.path.join(self.data_dir, cls)
            self.q = [cls_dir]  
            while len(self.q) > 0:
                path = self.q.pop()
                if os.path.isdir(path):
                    for i in os.listdir(path):
                        self.q.append(os.path.join(path, i))
                else:
                    img_paths.append((path, self.class_to_idx[cls]))
        return img_paths

    def get_img_paths_nolabel(self):
        img_paths = [] 
        q = [self.data_dir]

        while len(q) > 0:
            path = q.pop() 
            if os.path.isdir(path):
                for i in os.listdir(path):
                    q.append(os.path.join(path, i))
            else:
                img_paths.append((path, -1))
        return img_paths

    def shuffle(self): 
        np.random.shuffle(self.img_paths)
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx]
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label, img_path

