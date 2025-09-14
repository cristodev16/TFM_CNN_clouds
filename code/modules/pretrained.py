import torchvision.models as models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict
import numpy as np 
from difflib import SequenceMatcher
from sklearn.metrics import balanced_accuracy_score
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

    def __init__(self, model_name: str, pretrained: bool = False, lr: float = 1e-3, class_weights: list[float] | NoneType = None, device: torch.device | NoneType = None):
        self.model_name = model_name
        self.pretrained = pretrained
        self.initialize_model_weights()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.learning_rate = lr
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device)) if class_weights is not None else nn.CrossEntropyLoss()
        if device is None and "cuda" not in str(self.device):
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

    def _freeze_all_layers_but_fc(self):
        for param in self.model.parameters():
            param.requires_grad = False
        if "resnet" in self.model_name:
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif "vgg" in self.model_name:
            for param in self.model.classifier[-1].parameters():
                param.requires_grad = True
        elif "densenet" in self.model_name:
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def _unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def _reset_decision_layer(self, num_classes: int):
        if "resnet" in self.model_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif "vgg" in self.model_name:
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)
        elif "densenet" in self.model_name:
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def _set_optimizer(self, optimizer: str):
        if optimizer not in KnownModel.available_optimizers:
            suggested_optimizer = unmatch_suggest(KnownModel.available_optimizers, optimizer)
            raise ValueError(f"Selected optimizer not available. Did you mean {suggested_optimizer}? Available options are: {KnownModel.available_optimizers}. Using default (adam) now.")
        else:
            optimizers = {"adam": torch.optim.Adam((p for p in self.model.parameters() if p.requires_grad), self.learning_rate),
                          "sgd": torch.optim.SGD((p for p in self.model.parameters() if p.requires_grad), self.learning_rate), 
                          "rmsprop": torch.optim.RMSprop((p for p in self.model.parameters() if p.requires_grad), self.learning_rate)}
            self.optimizer = optimizers[optimizer]

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 400, patience: int = 30, validation: bool = True, early_stopping: bool = True) -> tuple[list[float], list[float], OrderedDict]:
        self._reset_decision_layer(num_classes=len(train_loader.dataset.classes))
        if self.pretrained:
            self._freeze_all_layers_but_fc()
        else:
            self._unfreeze_all()

        self._set_optimizer("adam")

        self.model.to(self.device)
        #best_model = deepcopy(self.model.state_dict())
        best_model = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        best_loss = float('inf')
        wait = 0 if early_stopping else None
        train_losses = []
        val_losses = [] if validation else None
        bal_acc_validation = None

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
                val_preds = []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        val_preds.extend(torch.max(outputs, 1)[1].cpu().numpy().tolist())
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                print(f"\t\tEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

                if early_stopping:
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        del best_model
                        #best_model = deepcopy(self.model.state_dict())
                        best_model = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                        bal_acc_validation = balanced_accuracy_score(val_loader.dataset.label_indices, val_preds)
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

        return train_losses, val_losses, n_epochs, self.model.state_dict(), bal_acc_validation
    
    def pred(self, test_loader: DataLoader) -> tuple[float, float, dict, np.ndarray]:
        self.model.to(self.device)
        self.model.eval()

        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_test_loss = test_loss / len(test_loader)

        print(f"\t\tTest Loss: {avg_test_loss:.4f}. Compute accuracy later with the predictions and labels.")
        return avg_test_loss, all_preds, all_labels
