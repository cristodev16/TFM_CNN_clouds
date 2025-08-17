import random
import numpy as np
import torch
import os 

def fix_seed(seed: int = 100510664):
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

def results_dir(model: str, pretrained: bool, main_rel_path: str = "../data/results"):
    """
    Creates a new folder to store results avoiding repetitons and overwritings of 
    previous similar verisons. Assumes fixed relative folders structure and existance
    of intermidate folders "data" and "results".

    Args:
        - model (str): Name of the model being trained
        - pretrained (bool): Whether the model is being trained fully or just the last 
        fully connected layer.

    Returns: 
         - str: Path created
    """
    model_path = model
    pretrained_path = "_pretrained" if pretrained else ""
    version = 1
    full_path = main_rel_path + "/" + model_path + pretrained_path + "_v" + str(version) + "/"
    while os.path.exists(full_path):
        version += 1
        full_path = main_rel_path + "/" + model_path + "_" + pretrained_path + "_v" + str(version) + "/"
    os.mkdir(full_path)
    return full_path
