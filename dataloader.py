import zipfile
import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

def unzip(filename):
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall("./")

class XRayDataset(Dataset):
    def __init__(self, stage='train', transform=None):
        self.path = f'./chest_xray/{stage}'
        self.stage = stage
        self.transform = transform
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        data = []
        labels = []
        for i in os.listdir(self.path + "/NORMAL"):
            if i.endswith('.jpeg'):
                data.append(self.path + "/NORMAL/" + i)
                labels.append(0)
        if self.stage == 'train':
            # Augument NORAML class with 2x data, since 1:3 ratio is not good
            for i in range(2):
                self.data.extend(data)
                self.labels.extend(labels)
        for i in os.listdir(self.path + "/PNEUMONIA"):
            if i.endswith('.jpeg'):
                self.data.append(self.path + "/PNEUMONIA/" + i)
                self.labels.append(1)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(self.data[idx])
        img = img.convert("L")
        if self.transform:
            img = self.transform(img)
            img /= 255
        return img, self.labels[idx]

class XRaySampleDataset(Dataset):
    def __init__(self, stage='train', sample_ratio=0.8, transform=None):
        self.path = f'./chest_xray/{stage}'
        self.transform = transform
        self.sample_ratio = sample_ratio
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for i in os.listdir(self.path + "/NORMAL"):
            if i.endswith('.jpeg'):
                self.data.append(self.path + "/NORMAL/" + i)
                self.labels.append(0)
        for i in os.listdir(self.path + "/PNEUMONIA"):
            if i.endswith('.jpeg'):
                self.data.append(self.path + "/PNEUMONIA/" + i)
                self.labels.append(1)
        shuffle = np.random.shuffle(zip(self.data, self.labels))
        self.data = [x for x, _ in shuffle]
        self.labels = [y for _, y in shuffle]
        self.data = self.data[:int(len(self.data) * self.sample_ratio)]
        self.labels = self.labels[:int(len(self.labels) * self.sample_ratio)]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(self.data[idx])
        img = img.convert("L")
        if self.transform:
            img = self.transform(img)
            img /= 255
        return img, self.labels[idx]

class XRayDataModule(pl.LightningDataModule):
    RESIZE_SHAPE = [224, 224]

    def __init__(self, batch_size=32, sample_ratio=None):
        super().__init__()
        self.batch_size = batch_size
        self.sample_ratio = sample_ratio


        transforms = Compose([
            Resize(self.RESIZE_SHAPE),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(15),
            ToTensor(),
        ])
        if self.sample_ratio is None:
            self.x_ray_train = XRayDataset(stage='train', transform=transforms)
            self.x_ray_val = XRayDataset(stage='val', transform=transforms)
            self.x_ray_test = XRayDataset(stage='test', transform=transforms)
        else:
            self.x_ray_train = XRaySampleDataset(stage='train', sample_ratio=self.sample_ratio, transform=transforms)
            self.x_ray_val = XRaySampleDataset(stage='val', sample_ratio=self.sample_ratio, transform=transforms)
            self.x_ray_test = XRaySampleDataset(stage='test', sample_ratio=self.sample_ratio, transform=transforms)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.x_ray_train, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.x_ray_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.x_ray_test, batch_size=self.batch_size, num_workers=4)

if __name__ == '__main__':
    unzip('archive.zip')