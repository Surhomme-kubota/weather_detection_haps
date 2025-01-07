import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset

class TrainDataset(BaseDataset):
    CLASSES = ['unlabel', 'tower']
    
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids = images_dir
        self.images_fps = images_dir
        self.masks_fps = masks_dir
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            image = self.preprocessing(image)
            mask = self.preprocessing(mask)
        return image, mask

    def __len__(self):
        return len(self.ids)

class TestDataset(BaseDataset):
    CLASSES = ['unlabel', 'tower']
    
    def __init__(self, images_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids = [str(images_dir)]
        self.images_fps = [str(images_dir)]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
        if self.preprocessing:
            image = self.preprocessing(image)
        return image

    def __len__(self):
        return len(self.ids)