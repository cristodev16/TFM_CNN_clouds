import torchvision.transforms as transforms
import numpy as np
from typing import Any

class pretrainedTransform(transforms.Compose):
    def __init__(self, transformations: list[Any] = [], resizing: bool = False):
        super().__init__(transformations)
        self.transforms = [transforms.Resize((224, 224)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        if not resizing:
            del self.transforms[0]

class customTransform(transforms.Compose):
    def __init__(self, images_to_normalize: np.ndarray, transformations: list[Any] = [], resizing: bool = False):
        super().__init__(transformations)
        self.transforms = [transforms.Resize((224, 224)),
                           transforms.ToTensor()]
        if not resizing:
            del self.transforms[0]
            
        images_to_normalize = images_to_normalize / 255.0 if np.max(images_to_normalize) > 1 else images_to_normalize
        mean = np.mean(images_to_normalize, axis=(1,2,3))
        std = np.std(images_to_normalize, axis=(1,2,3))
        self.transforms.append(transforms.Normalize(mean=mean, std = std))
        
        


        