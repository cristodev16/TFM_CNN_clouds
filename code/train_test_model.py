import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from modules.transformations import pretrainedTransforms
from modules.data import Data
from modules.pretrained import Pretrained
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
    labels_df_path = "/home/csanchezmoreno/tfm/data/metadata_reduced.pickle"
    images_path = "/home/csanchezmoreno/tfm/data/imageset_reduced.pickle"
    data = Data(images_path, labels_df_path)

    # VARIABLES AND MODEL INITIALIZATION AND CONFIGURATION
    transform = pretrainedTransforms() # Only one fixed option (pretrained for now) for the transform
    train_all_data = True
    chosen_model = "resnet18"
    model = Pretrained(chosen_model)
    model.reset_fc_layer()
    model.freeze_all_layers_but_fc()
    ns_epochs = []
    

    # Iterative training and testing 
    for i in range(len(test_dates)+1):
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed+i)
        if i != len(test_dates):
            # Here we train and evaluate for each testing partition. We have to do it twice: Once with validation to retrieve 
            # the optimal number of epochs and one without validation using that number of epochs and the whole training dataset 

            train_loader, train_train_loader, val_loader, test_loader = data.get_loaders(test_dates[i], stratified_split, transform)








            # AFTER TRAINING AND EVALUATING DON'T FORGET TO FREE BOTH GPU AND CPU RESOURCES TO 
            # AVOID OUT_OF_MEMORY EXCEPTIONS 




        elif train_all_data:
            # Here we train with the whole dataset using a number of epochs being the average of the optimal ones found before
            # and we keep as estimation of future performance the average of the performance measures obtained for each testing 
            # partition 
            pass

if __name__=="__main__":
    main()