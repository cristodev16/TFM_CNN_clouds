import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from modules.transformations import pretrainedTransforms
from torch.utils.data import DataLoader
from modules.data import MyDataset
import argparse

def fix_seed(seed=42):
    """
    Fix the seed for multiple random modules in different used (or potentially used) 
    libraries. Forces deterministic behaviour in the results at the GPU hardware 
    level as well.

    Args:
        - seed (int).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    Main workflow execution function.
    """
    seed = 100510664
    fix_seed(seed)

    test_dates = pd.read_csv("../data/test_dates.csv")["test_datetimes"].tolist()
    labels_df = pd.read_pickle("../data/metadata_reduced.pickle").reset_index()
    images = pd.read_pickle("../data/images_reduced.pickle")

    # Only one fixed option (for now) for the transform
    transform = pretrainedTransforms()
   
    for i in range(len(test_dates)+1):
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed+i)
        if i != len(test_dates):
            # Here we train and evaluate for each testing partition. We have to do it twice: Once with validation to retrieve 
            # the optimal number of epochs and one without validation using that number of epochs and the whole training dataset 

            # GATHER THE TRAINING AND TESTING DATA (VALIDATION AND TRAINING BSAED ON RANDOM SHUFFLED STRATIFIED SPLIT). MAYBE
            # EMBBED INTO A FUNCTION FOR BETTER MEMORY MANAGEMENT AND READABILITY 
            test_indices = np.array(labels_df[labels_df["datetimes"].dt.date == pd.to_datetime(test_dates[i]).date()].index.tolist())
            train_indices = np.array(labels_df[labels_df["datetimes"].dt.date != pd.to_datetime(test_dates[i]).date()].index.tolist())

            train_labels = labels_df["types"].iloc[train_indices].values
            train_train_idx, train_val_idx = next(stratified_split(np.arange(len(train_labels)), train_labels))

            train_train_indices = train_indices[train_train_idx]
            train_val_indices = train_indices[train_val_idx]

            train_dataset = MyDataset(images=images[..., train_train_indices], labels_df=labels_df.iloc[train_train_indices], transform=transform)
            val_dataset = MyDataset(images=images[..., train_val_indices], labels_df=labels_df.iloc[train_val_indices], transform=transform)
            test_dataset = MyDataset(images=images[..., test_indices], labels_df=labels_df.iloc[test_indices], transform=transform)

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            train_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
            train_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)



            # AFTER TRAINING AND EVALUATING DON'T FORGET TO FREE BOTH GPU AND CPU RESOURCES TO 
            # AVOID OUT_OF_ERROR EXCEPTIONS 




        else:
            # Here we train with the whole dataset using a number of epochs being the average of the optimal ones found before
            # and we keep as estimation of future performance the average of the performance measures obtained for each testing 
            # partition 
            pass







if __name__=="__main__":
    main()