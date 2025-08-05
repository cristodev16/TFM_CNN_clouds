from typing import Any
from torch.utils.data import Dataset
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
