import torchvision.models as models
from difflib import SequenceMatcher
from types import NoneType
from copy import deepcopy
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader

def unmatch_suggest(available_elements: list[str], element: str) -> NoneType | str:
    similarity = 0
    for available_element in available_elements:
        new_similarity = SequenceMatcher(None, available_element, element).ratio()
        if new_similarity > similarity:
            similarity = new_similarity
            suggested = available_element
    return suggested if suggested else None

class pretrained:
    available_optimizers = ["adam", "sgd", "rmsprop"]

    def __init__(self, model_name: str, device: torch.device | NoneType = None, lr: float | NoneType = None):
        self.select_model_initialize_weights(model_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.learning_rate = lr if lr is not None else 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def select_model_initialize_weights(self, model_name: str):
        if model_name in models.list_models():
            try:
                self.model_weights = models.get_model_weights(model_name).DEFAULT
            except Exception:
                raise ValueError("Chosen model does not have existing pre-trained weights in torchvision package.")
        else:
            suggested_model = unmatch_suggest(models.list_models(), model_name)
            if suggested_model:
                raise ValueError(f"Input model name does not match any of the available ones in torch vision. Did you mean {suggested_model}?")
            else:
                raise ValueError(f"Input model name does not match any of the available ones in torch vision.")
        self.model = models.get_model(model_name, weights=self.model_weights)

    def freeze_all_layers_but_fc(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param_fc in self.model.fc.parameter():
            param_fc.requires_grad = True

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def reset_fc_layer(self, num_classes: int):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def set_optimizer(self, optimizer: str):
        if optimizer not in pretrained.available_optimizers:
            suggested_optimizer = unmatch_suggest(pretrained.available_optimizers, optimizer)
            print(f"Selected optimizer not available. Did you mean {suggested_optimizer}? Available options are: {pretrained.available_optimizers}. Using default (adam) now.")
        else:
            optimizers = {"adam": torch.optim.Adam(self.model.parameters(), self.learning_rate),
                          "sgd": torch.optim.SGD(self.model.parameters(), self.learning_rate), 
                          "rmsprop": torch.optim.RMSprop(self.model.parameters(), self.learning_rate)}
            self.optimizer = optimizers[optimizer]

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 30, patience: int = 5, validation: bool = True, early_stopping: bool = True) -> tuple[list[float], list[float], OrderedDict]:
        self.model.to(self.device)
        best_model = deepcopy(self.model.state_dict())
        best_loss = float('inf')
        wait = 0 if early_stopping else None
        train_losses = []
        val_losses = [] if validation else None

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            if not validation: 
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

            if validation:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

                if early_stopping:
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        del best_model
                        best_model = deepcopy(self.model.state_dict())
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            n_epochs = epoch+1
                            print(f"Early stopping at epoch {n_epochs}")
                            break
        if validation and early_stopping: 
            self.model.load_state_dict(best_model)
        else:
            n_epochs = epochs

        return train_losses, val_losses, n_epochs, self.model.state_dict()
    
    def test(self):
        pass
