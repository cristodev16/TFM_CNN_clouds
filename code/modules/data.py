from typing import Any
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from modules.transformations import pretrainedTransforms
import numpy as np
from PIL import Image
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, images: np.ndarray, labels_df: pd.DataFrame, transform: Any = None):
        self.images = np.transpose(images, (3, 0, 1, 2))  # (N, C, H, W)
        self.labels = labels_df["types"].values
        self.transform = transform

        self.classes = sorted(set(self.labels))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.label_indices = np.array([self.class_to_idx[l] for l in self.labels])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]  # (C, H, W)
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        img = Image.fromarray((img * 255).astype(np.uint8)) if img.max() <= 1 else Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        label = self.label_indices[idx]
        return img, label

class Data:
    def __init__(self, path_to_images: str, path_to_df: str):
        self.path_to_df = path_to_df
        self.path_to_images = path_to_images
        self.data_df = None
        self.data_images = None

    def get_train_test_val_indices(self, date: str, stratified_split: StratifiedShuffleSplit) -> tuple[np.ndarray]:
        if self.data_df is None:
            self.data_df = pd.read_pickle(self.path_to_df).reset_index()
        test_indices = np.array(self.data_df[self.data_df["datetimes"].dt.date == pd.to_datetime(date).date()].index.tolist())
        train_indices = np.array(self.data_df[self.data_df["datetimes"].dt.date != pd.to_datetime(date).date()].index.tolist())

        train_labels = self.data_df["types"].iloc[train_indices].values
        train_train_idx, train_val_idx = next(stratified_split.split(np.arange(len(train_labels)), train_labels))

        train_train_indices = train_indices[train_train_idx]
        train_val_indices = train_indices[train_val_idx]

        return train_indices, train_train_indices, train_val_indices, test_indices
    
    def get_loaders(self, date: str, stratified_split: StratifiedShuffleSplit, transformation: pretrainedTransforms, batch_sizes: tuple[int] = (32,32,8)) -> tuple[DataLoader]:
        train_indices, train_train_indices, train_val_indices, test_indices = self.get_train_test_val_indices(date=date, stratified_split=stratified_split)
        if self.data_images is None:
            self.data_images = pd.read_pickle(self.path_to_images)

        train_train_dataset = MyDataset(images=self.data_images[..., train_train_indices], labels_df=self.data_df.iloc[train_train_indices], transform=transformation)
        train_dataset = MyDataset(images=self.data_images[..., train_indices], labels_df=self.data_df.iloc[train_indices], transform=transformation)
        val_dataset = MyDataset(images=self.data_images[..., train_val_indices], labels_df=self.data_df.iloc[train_val_indices], transform=transformation)
        test_dataset = MyDataset(images=self.data_images[..., test_indices], labels_df=self.data_df.iloc[test_indices], transform=transformation)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True)
        train_train_loader = DataLoader(train_train_dataset, batch_size=batch_sizes[0], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_sizes[1], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_sizes[2], shuffle=True)

        return train_loader, train_train_loader, val_loader, test_loader
    
    def get_full_loader(self, transformation: pretrainedTransforms, batch_size: int = 64):
        dataset = MyDataset(images=self.data_images, labels_df=self.data_df, transform=transformation)
        return DataLoader(dataset, batch_size=batch_size)
        
