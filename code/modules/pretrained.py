import torchvision.models as models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np 
from difflib import SequenceMatcher
from types import NoneType

def unmatch_suggest(available_elements: list[str], element: str) -> NoneType | str:
    similarity = 0
    for available_element in available_elements:
        new_similarity = SequenceMatcher(None, available_element, element).ratio()
        if new_similarity > similarity:
            similarity = new_similarity
            suggested = available_element
    return suggested if suggested else None

class KnownModel:
    available_optimizers: list[str] = ["adam", "sgd", "rmsprop"]

    def __init__(self, model_name: str, pretrained: bool = False, device: torch.device | NoneType = None, lr: float | NoneType = None):
        self.model_name = model_name
        self.pretrained = pretrained
        self.initialize_model_weights()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.learning_rate = lr if lr else 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        if device is None and self.device == "cpu":
            print("WARNING: GPU not found to be available. Model and data will be loaded into CPU.")

    def initialize_model_weights(self):
        if self.model_name in models.list_models():
            if self.pretrained:
                try:
                    self.model_weights = models.get_model_weights(self.model_name).DEFAULT
                except Exception:
                    raise ValueError("Chosen model does not have existing pre-trained weights in torchvision package.")
            else:
                self.model_weights = None
        else:
            suggested_model = unmatch_suggest(models.list_models(), self.model_name)
            if suggested_model:
                raise ValueError(f"Input model name does not match any of the available ones in torch vision. Did you mean {suggested_model}?")
            else:
                raise ValueError(f"Input model name does not match any of the available ones in torch vision.")
        self.model = models.get_model(self.model_name, weights=self.model_weights)

    def freeze_all_layers_but_fc(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param_fc in self.model.fc.parameters():
            param_fc.requires_grad = True

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def reset_fc_layer(self, num_classes: int):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def set_optimizer(self, optimizer: str):
        if optimizer not in KnownModel.available_optimizers:
            suggested_optimizer = unmatch_suggest(KnownModel.available_optimizers, optimizer)
            print(f"WARNING: Selected optimizer not available. Did you mean {suggested_optimizer}? Available options are: {KnownModel.available_optimizers}. Using default (adam) now.")
        else:
            optimizers = {"adam": torch.optim.Adam(self.model.parameters(), self.learning_rate),
                          "sgd": torch.optim.SGD(self.model.parameters(), self.learning_rate), 
                          "rmsprop": torch.optim.RMSprop(self.model.parameters(), self.learning_rate)}
            self.optimizer = optimizers[optimizer]

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 20, patience: int = 5, validation: bool = True, early_stopping: bool = True) -> tuple[list[float], list[float], OrderedDict]:
        self.reset_fc_layer(num_classes=len(train_loader.dataset.classes))
        if self.pretrained:
            self.freeze_all_layers_but_fc()
        else:
            self.unfreeze_all()

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

                print(f"\t\tEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

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

                print(f"\t\tEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

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
                            print(f"\t\tEarly stopping at epoch {n_epochs} (patience {patience})...")
                            n_epochs = n_epochs - patience
                            break
        else:
            n_epochs = epochs
        if validation and early_stopping: 
            self.model.load_state_dict(best_model)

        return train_losses, val_losses, n_epochs, self.model.state_dict()
    
    def test(self, test_loader: DataLoader) -> tuple[float, float, dict, np.ndarray]:
        self.model.to(self.device)
        self.model.eval()

        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct / total
        class_report = classification_report(all_labels, all_preds, output_dict=True)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        print(f"\t\tTest Loss: {avg_test_loss:.4f} - Test Accuracy: {accuracy*100:.2f}%")
        return avg_test_loss, accuracy, class_report, conf_matrix
