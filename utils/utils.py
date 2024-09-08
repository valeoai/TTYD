import os

import numpy as np
import scipy
import torch
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import utils.metrics as metrics
from lightconvpoint.utils.misc import dict_to_device


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC
def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC
def wred(str):
    return bcolors.FAIL+str+bcolors.ENDC

def get_savedir_root(config, experiment_name):
    
    savedir = f"{experiment_name}_{config['network_decoder']}_{config['network_decoder_k']}"
    savedir += f"_{config['train_split']}Split"
    if ("desc" in config) and config["desc"]:
        savedir += f"_{config['desc']}"


    print(savedir)
    savedir_root = os.path.join(config["save_dir"], savedir)

    return savedir_root

def get_bbdir_root(config):
    # Gives the backbone dir 
    savedir_root = config["da_fixed_head_path_model"]

    return savedir_root