import torchvision.transforms as transforms
from typing import Any

class pretrainedTransforms(transforms.Compose):
    def __init__(self, transformations: list[Any] = [], scaling: bool = True):
        super().__init__(transformations)
        self.transforms = [transforms.Resize((224, 224)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        if scaling == False:
            del self.transforms[0]
        