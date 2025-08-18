import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from modules.transformations import pretrainedTransforms
from modules.data import Data
from modules.pretrained import KnownModel
from modules.tools import fix_seed, results_dir
from modules.decorators import output_state_warning
import argparse
from argparse import Namespace
import json
import time
import os 

def main():
    """
    Program execution function. Pre-loads some data and parses the arguments that uses later 
    to execute the main logic in sub_main() function, to which these arguments are passed through
    a warning decorator.
    """

    # PARSE ARGUMENTS
    argparser = argparse.ArgumentParser(description="Trainer/Tester of multiple pretrained models' architectures.")
    argparser.add_argument("-m", "--model", required=True, help="Indicate the model (available in pytorch) whose architecture we will use.")
    argparser.add_argument("-p", "--pretrained", action="store_true", help="Activate to train dense layer only for the selected model")
    argparser.add_argument("-r", "--resize", action="store_true", help="Avtivate to use resizing in the transformer to adapt the input images' sizes to those of ImageNet.")
    argparser.add_argument("-s", "--simplified_classes", action="store_true", help="Avtivate to use use a simplified set of classes for our problem.")
    argparser.add_argument("-ft", "--full_train", action="store_true", help="Activate to train the selected parts of the chosen model using all the available data once all testing has been done.")
    args = argparser.parse_args()

    # LOAD PRE-DATA
    test_dates = pd.read_csv("../data/test_dates.csv")["test_datetimes"].tolist()
    save_dir = results_dir(args.model, args.pretrained, args.simplified_classes)

    # Execute main logic of the program
    sub_main(args, test_dates, save_dir)

@output_state_warning
def sub_main(args: Namespace, test_dates: list[str], save_dir: str):
    """
    Main logic of the program.
    """

    print("PROGRAM INITIALIZED: Trainining and testing model with the following configuration: \n"
        f"\t- Model: {args.model}\n"
        f"\t- Pretrained: {args.pretrained}\n"
        f"\t- Resize: {args.resize}\n"
        f"\t- Simplified classes: {args.simplified_classes}\n"
        f"\t- Test-set days: {test_dates}\n"
        f"\t- Saving directory: {save_dir}\n"
        f"\t- Full Train: {args.full_train} \n\n")

    print("Initializing log: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
    print("- Initializing variables and model...")

    seed = 100510664
    fix_seed(seed)

    # LOAD DATA 
    labels_df_path = "/home/csanchezmoreno/tfm/data/metadata_reduced.pickle"
    images_path = "/home/csanchezmoreno/tfm/data/imageset_reduced.pickle"
    data = Data(images_path, labels_df_path, simplified=args.simplified_classes)

    # VARIABLES AND MODEL INITIALIZATION AND CONFIGURATION
    transform = pretrainedTransforms(resizing=args.resize) # Only one fixed option (pretrained for now) for the transform
    model = KnownModel(args.model, args.pretrained) # So far, fixed layers selection to train based on whether we choose pretrained or not.
    results = {}

    print("- Initializing the training/testing...")

    # Iterative training and testing 
    for i in range(len(test_dates)+1):
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed+i)
        if i != len(test_dates):
            print(f"\t- Testing date: {test_dates[i]}")

            print("\t\t- Loading required data...")
            # Get DataLoader() objects
            train_loader, train_train_loader, val_loader, test_loader = data.get_loaders(test_dates[i], stratified_split, transform)

            print("\t\t- Training with validation for number of epochs adjustment...")
            # Train with validation and early_stopping
            train_train_losses, val_losses, n_epochs, parameters_val = model.train(train_loader=train_train_loader, val_loader=val_loader)
            results[test_dates[i]] = {"train_train_losses": train_train_losses, "val_losses": val_losses, "n_epochs": n_epochs}
            print(f"\t\tSuccessful training: Saving model...")
            torch.save(parameters_val, save_dir + f"model_weights_{test_dates[i]}_validation.pth")
            torch.cuda.empty_cache()
            
            print(f"\t\t- Training with whole training data and {n_epochs} epochs...")
            # Re-initialize model and train with a fixed number of epochs
            model.initialize_model_weights()
            train_losses, _, _, parameters = model.train(train_loader=train_loader, epochs=n_epochs, validation=False)
            results[test_dates[i]]["train_losses"] = train_losses
            print(f"\t\tSuccessful training: Saving model...")
            torch.save(parameters, save_dir + f"model_weights_{test_dates[i]}.pth")
            torch.cuda.empty_cache()

            print("\t\t- Testing and extracting measures...")
            # Test model and keep results
            avg_test_loss, accuracy, class_report, conf_matrix = model.test(test_loader=test_loader)
            results[test_dates[i]]["avg_test_loss"] = avg_test_loss
            results[test_dates[i]]["accuracy"] = accuracy
            results[test_dates[i]]["class_report"] = class_report
            results[test_dates[i]]["cm"] = conf_matrix.tolist()

            print("\tProcess successful: Cleaning data...")
            # Clean loaders and re-initialize weights
            torch.cuda.empty_cache()
            del train_train_loader, train_loader, val_loader, test_loader
            model.initialize_model_weights()

        elif args.full_train:
            print(f"\t- Retraining the with the whole dataset and avergaed number of epochs...")

            # Get average number of epochs
            epochs = [results_i["n_epochs"] for _, results_i in results.items()]
            avg_epochs = int(np.round(np.average(epochs)))

            print("\t\t- Loading required data...")
            # Get dataloader with all information 
            dataloader = data.get_full_loader(transformation=transform)

            print("\t\t- Training...")
            # Reinitialize model, train and save data
            model.initialize_model_weights()
            losses, _, _, full_parameters = model.train(train_loader=dataloader, epochs=avg_epochs, validation=False)
            print(f"\t\tSuccessful training: Saving model...")
            results["full_model"] = {"train_losses": losses}
            torch.save(full_parameters, save_dir + f"full_model_weights.pth")

            print("\tProcess successful: Cleaning data...")
            torch.cuda.empty_cache()
            del dataloader, data, model

    print(f"- Saving result metrics...\n\n")
    with open(save_dir+"results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("PGROGRAM FINISHED: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")

if __name__=="__main__":
    main()