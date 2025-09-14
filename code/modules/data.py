from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from modules.transformations import pretrainedTransform, customTransform
from modules.tools import get_weights
import numpy as np
from PIL import Image
import pandas as pd
import json

class MyDataset(Dataset):
    def __init__(self, trained_encoder: LabelEncoder, images: np.ndarray, labels_df: pd.DataFrame, transform: transforms.Compose = None):
        self.images = np.transpose(images, (3, 0, 1, 2))  # (N, C, H, W)
        self.labels = labels_df["types"].values
        self.classes = trained_encoder.classes_
        self.label_indices = trained_encoder.transform(self.labels)
        self.transform = transform

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
    def __init__(self, path_to_images: str, path_to_df: str, simplified: bool = False, class_weights: bool = False, pretrained: bool = False, resize: bool = False):
        self.data_df = pd.read_pickle(path_to_df).reset_index()
        self.data_images = pd.read_pickle(path_to_images) # (C, H, W, N)
        if simplified:
            replacements = {"cirrocumulo": "cirro/alto-cúmulo", "altocumulo": "cirro/alto-cúmulo",
                            "cirro": "cirro(estrato)", "cirroestrato": "cirro(estrato)",
                            "altoestrato": "(alto)estrato", "estrato": "(alto)estrato"}
            self.data_df["types"] = self.data_df["types"].replace(replacements)
        self.label_encoder = LabelEncoder()
        classes = sorted(set(self.data_df["types"].values))
        self.label_encoder.fit(classes)
        self.inverse_freq = torch.tensor(get_weights(classes, self.data_df["types"].values.tolist()), dtype=torch.float) if class_weights else None
        self.pretrained = pretrained
        self.resize = resize

    def _get_train_test_val_indices(self, date: str | list[str], stratified_split: StratifiedShuffleSplit | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        test_indices_list = []
        train_indices_list = []
        if isinstance(date, str):
            date = [date]
        for day in date:
            test_indices_list.extend(self.data_df[self.data_df["datetimes"].dt.date == pd.to_datetime(day).date()].index.tolist())
        train_indices_list = [idx for idx in self.data_df.index.tolist() if idx not in test_indices_list]

        test_indices = np.array(test_indices_list)
        train_indices = np.array(train_indices_list)

        if stratified_split is not None:
            train_labels = self.data_df["types"].iloc[train_indices].values
            train_train_idx, train_val_idx = next(stratified_split.split(np.arange(len(train_labels)), train_labels))

            train_train_indices = train_indices[train_train_idx]
            train_val_indices = train_indices[train_val_idx]

            return train_indices, train_train_indices, train_val_indices, test_indices
        else:
            train_df = self.data_df.iloc[train_indices]
            train_val_indices = []
            selected_days = []
            picked_classes = []
            forbidden_days = []
            for label in self.label_encoder.classes_:
                temp_class_df = train_df[train_df["types"]==label]
                days_per_class_count = temp_class_df.groupby(temp_class_df["datetimes"].dt.date).size().reset_index(name="count").sort_values(by="count", ascending=True)
                if len(days_per_class_count) > 1 and label not in picked_classes:
                    if len(days_per_class_count) == 2:
                        selected_days.extend(days_per_class_count.iloc[0:1,0].tolist())
                    else:    
                        selected_days.extend(days_per_class_count.iloc[0:2,0].tolist())
                    picked_classes.append(label)
                else:
                    forbidden_days.append(days_per_class_count.iloc[0,0])
            selected_days = list(set([day for day in selected_days if day not in forbidden_days]))
            for day in selected_days:
                train_val_indices.extend(train_df[train_df["datetimes"].dt.date == day].index.tolist())
            train_val_indices = np.array(train_val_indices)
            train_train_indices = np.array(train_df.drop(train_val_indices).index.tolist())

            return train_indices, train_train_indices, train_val_indices, test_indices
    
    def get_loaders(self, date: str, validation: bool = True, stratified_split: StratifiedShuffleSplit | None = None, batch_sizes: tuple[int] = (12,12,8)) -> tuple[DataLoader, DataLoader | None, DataLoader | None, DataLoader]:
        date = json.loads(date.replace("'", '"')) if "[" in date else date
        train_indices, train_train_indices, train_val_indices, test_indices = self._get_train_test_val_indices(date=date, stratified_split=stratified_split)

        if validation:
            transformation = pretrainedTransform(resizing=self.resize) if self.pretrained else customTransform(resizing=self.resize, images_to_normalize = self.data_images[..., train_train_indices])

            train_train_dataset = MyDataset(images=self.data_images[..., train_train_indices], labels_df=self.data_df.iloc[train_train_indices], transform=transformation, trained_encoder=self.label_encoder)
            val_dataset = MyDataset(images=self.data_images[..., train_val_indices], labels_df=self.data_df.iloc[train_val_indices], transform=transformation, trained_encoder=self.label_encoder)
            train_train_loader = DataLoader(train_train_dataset, batch_size=batch_sizes[0], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_sizes[1], shuffle=True)
            
        else:
            transformation = pretrainedTransform(resizing=self.resize) if self.pretrained else customTransform(resizing=self.resize, images_to_normalize = self.data_images[..., train_indices])

            train_train_loader = None
            val_loader = None
        
        train_dataset = MyDataset(images=self.data_images[..., train_indices], labels_df=self.data_df.iloc[train_indices], transform=transformation, trained_encoder=self.label_encoder)
        test_dataset = MyDataset(images=self.data_images[..., test_indices], labels_df=self.data_df.iloc[test_indices], transform=transformation, trained_encoder=self.label_encoder)
        train_loader = DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_sizes[2], shuffle=True)

        return train_loader, train_train_loader, val_loader, test_loader
    
    def get_full_loader(self, batch_size: int = 12):
        transformation = pretrainedTransform(resizing=self.resize) if self.pretrained else customTransform(resizing=self.resize, images_to_normalize = self.data_images)
        dataset = MyDataset(images=self.data_images, labels_df=self.data_df, transform=transformation, trained_encoder = self.label_encoder)
        return DataLoader(dataset, batch_size=batch_size)
        
