import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import pandas as pd 
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, confusion_matrix
import json
import argparse
import os
import shutil

# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("-p", "--path", required=True, help="Takes the path where the file results.json to process will be stored and where the plots and metrics will be saved.")
args = argparser.parse_args()

# Get full path to the file and generate folder to store parsed results
path_to_folder = args.path if args.path.endswith("/") else args.path + "/"
path_results = path_to_folder + "parsed_results/"
if os.path.isdir(path_results):
    shutil.rmtree(path_results)
os.mkdir(path_results)

# Load the results file
with open(path_to_folder + "results.json") as f:
    results = json.load(f)

# Get all losses plots and save them in the parsed results folder
for key in results:
    if "2015" in key:
        x = [i+1 for i in range(len(results[key]["val_losses"]))]
        x_short = [i+1 for i in range(results[key]["n_epochs"])]
        plt.figure()
        plt.plot(x, results[key]["val_losses"], label="Validation set", color="#0085fa")
        plt.plot(x, results[key]["train_train_losses"], label="Train_train set", color="#910000", linestyle="--")
        if "train_losses" in results[key]:
            plt.plot(x_short, results[key]["train_losses"], label="Full train set", color="#4f8700", linestyle="-.")
        plt.xlabel("Number of epochs", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(loc="upper right", fontsize=14)
        plt.savefig(path_results + key + "_loss_plots.png")

# Compute evaluation metrics
# 1. Concatenate all predictions and real labels (coming from multiple evaulations)
y_pred_total = []
y_true_total = []
for key in results:
    if "2015" in key:
        y_pred_total.extend(results[key]["predictions"])
        y_true_total.extend(results[key]["labels"])
y_pred_total = np.array(y_pred_total)
y_true_total = np.array(y_true_total)

# 2. Get label names in the order of the encoded variables:
labels = [key for key, _ in results["encoding_map"].items()]

# 3. Compute per-class metrics and save them into a csv
p, r, f1, support = precision_recall_fscore_support(y_true_total, y_pred_total, zero_division=0)
per_class_metrics = pd.DataFrame({"Class": labels, "Precision": p, "Recall": r, "F1-Score": f1, "Support":support})
per_class_metrics.to_csv(path_results+"per_class_metrics.csv")

# 4. Compute some overall metrics and save them into a json
overall_metrics = {"accuracy": accuracy_score(y_true_total, y_pred_total),
                   "balanced_accuracy": balanced_accuracy_score(y_true_total, y_pred_total),
                   "macro_precision": p.mean(),
                   "macro_recall": r.mean(),
                   "macro_f1": f1.mean()}
with open(path_results + "overall_metrics.json", "w") as f:
    json.dump(overall_metrics, f, indent=4)

# 5. Compute confusion matrix and save as json (list) and as figure
cm = confusion_matrix(y_true_total, y_pred_total).tolist()
with open(path_results + "confusion_matrix.json", "w") as f:
    json.dump(cm, f)



# 6. Create summary text in .txt file
summary_path = path_results + "summary_metrics.txt"
with open(summary_path, "w") as f:
    f.write("==== Overall Metrics ====\n")
    for k, v in overall_metrics.items():
        f.write(f"{k:20s}: {v:.4f}\n")

    f.write("\n==== Per-Class Metrics ====\n")
    f.write(per_class_metrics.to_string(index=False))
    f.write("\n")

    f.write("\n==== Confusion Matrix ====\n")
    f.write("Rows = True labels | Cols = Predicted labels\n")
    f.write(np.array2string(np.array(cm), separator=", "))
