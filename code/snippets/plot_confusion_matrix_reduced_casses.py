#!/usr/bin/env python3
import argparse
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Plot confusion matrix from json.")
    parser.add_argument("-p", "--path", type=str, help="Path to folder containing confusion_matrix.json")
    args = parser.parse_args()

    json_file = os.path.join(args.path, "confusion_matrix.json")
    output_dir = args.path
    output_file = os.path.join(output_dir, "confusion_matrix.png")

    # Load confusion matrix data
    with open(json_file, "r") as f:
        cm = json.load(f)

    # Fixed class labels. Consider that this is the labels order in which data was generated.
    # Check metrics summary at files summary_metrics.txt 
    labels = [
        "Stratus + Altostratus",
        "Clear Sky",
        "Cirrus + Cirrostratus",
        "Cirrocumulus + Altocumulus",
        "Cumulus",
        "Stratocumulus"
    ]

    # Plot
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels)
    #plt.title("Confusion Matrix", pad=20, fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=12) # Rotate labels in x axis for them to fit
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Confusion matrix plot saved at: {output_file}")

if __name__ == "__main__":
    main()
