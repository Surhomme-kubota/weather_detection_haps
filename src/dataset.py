from torch.utils.data import Dataset
from glob import glob
import os
from PIL import Image
from pathlib import Path

BASEPATH = Path(__file__).parent.parent

# train_Dataset
class TrainDataset(Dataset):
    def __init__(self, transforms=None):
        self.dirs = glob(str(BASEPATH / 'data' / 'treated_image' / 'train' / '*' /'*.jpg'))
        self.labels = []
        for dir in self.dirs:
            label = int(dir.split('/')[-2])
            self.labels.append(label)
        self.transform = transforms
    def __getitem__(self,idx):
        im = Image.open(self.dirs[idx]).convert("RGB")
        self.im = self.transform(im)
        return self.im, self.labels[idx]
    def __len__(self):
        return len(self.dirs)
    

# test_dataset
class TestDataset(Dataset):
    def __init__(self, transforms=None):
        self.dirs = glob(str(BASEPATH / 'data'/ 'treated_image' / 'val' / '*' /'*.jpg'))
        self.labels = []
        for dir in self.dirs:
            label = int(dir.split('/')[-2])
            self.labels.append(label)
        self.transform = transforms
    def __getitem__(self,idx):
        im = Image.open(self.dirs[idx]).convert("RGB")
        self.im = self.transform(im)
        return self.im, self.labels[idx]
    def __len__(self):
        return len(self.dirs)
    
    
class EvalDataset(Dataset):
    def __init__(self, transforms=None):
        self.dirs = glob(str(BASEPATH / 'data'/ 'treated_image' / '*.jpg'))
        self.labels = []
        for dir in self.dirs:
            try:
                label = int(dir.split('/')[-2])
                self.labels.append(label)
            except:
                self.labels.append(0)
        self.transform = transforms

    def __getitem__(self, idx):
        im = Image.open(self.dirs[idx]).convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, self.labels[idx], self.dirs[idx]

    def __len__(self):
        return len(self.dirs)