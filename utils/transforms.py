import importlib
import logging
import math
import numbers
import random
import re
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

import lightconvpoint.utils.transforms as lcp_T

from .all_transforms import *

torchsparse_found = importlib.util.find_spec("torchsparse") is not None
logging.info(f"Torchsparse found - {torchsparse_found}")
if torchsparse_found:
    from torchsparse import SparseTensor
    from torchsparse.utils.quantize import sparse_quantize

me_found = importlib.util.find_spec("MinkowskiEngine") is not None
logging.info(f"ME found - {torchsparse_found}")
warnings.filterwarnings("ignore", message="__floordiv__ is deprecated, and its behavior will change in a future version of pytorch")
logging.info("Filtering in TorchSparseQuantize the warning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch")


def da_get_inputs(config):
    
    # compute the input size:
    in_channels_source = 0
    inputs_source = []
    in_channels_target = 0
    inputs_target = []
    if config["source_input_intensities"]:
        inputs_source.append("intensities")
        in_channels_source += 1
    if config["source_input_dirs"]:
        in_channels_source += 3
        inputs_source.append("dirs")
    if in_channels_source == 0:
        in_channels_source += 1
        inputs_source.append("x")
    
    if config["target_input_intensities"]:
        inputs_target.append("intensities")
        in_channels_target += 1
    if config["target_input_dirs"]:
        in_channels_target += 3
        inputs_target.append("dirs")
    if in_channels_target == 0:
        in_channels_target += 1
        inputs_target.append("x")

    return in_channels_source, inputs_source, in_channels_target, inputs_target

def da_get_transforms(config, network_function=None, train=True, downstream=False, keep_orignal_data=False, source_flag=True):
    logging.info(f"Transforms - Train {train} - Downstream {downstream}")
    _, inputs_source, _, inputs_target = da_get_inputs(config)
    if source_flag:
        #If it is SOURCE
        inputs = inputs_source
    else: 
        #If it is TARGET
        inputs = inputs_target

    transforms = []
    if keep_orignal_data:
        logging.info("Transforms - Dupplicate - pos, y")
        transforms.append(Dupplicate(["pos", "y"], "original_"))
    logging.info(f"Transforms - No voxel decimation")



    if train and not config['no_augmentation']:
        if ("randRotationZ" in config) and config["randRotationZ"]:
            item_list = ["pos"]
            logging.info(f"Transforms - Rotation on: {item_list}")
            transforms.append(RandomRotate(180, axis=2, item_list=item_list))

        if ("randFlip" in config) and config["randFlip"]:
            item_list = ["pos"]
            logging.info("Transforms - Flip on {}".format(item_list))
            transforms.append(RandomFlip(item_list))
    
    logging.info("Transforms - CreateInputs")
    transforms.append(CreateInputs(inputs))
    
    if "TorchSparse" in config["network_backbone"]:
        if keep_orignal_data:
            item_list = ['pos', 'original_pos']
            logging.info(f"Transforms - TorchSparseQuantize on {item_list}")
            transforms.append(TorchSparseQuantize(config["voxel_size"], item_list))

        else:
            item_list = ['pos']
            logging.info(f"Transforms - TorchSparseQuantize on {item_list}")
            transforms.append(TorchSparseQuantize(config["voxel_size"], item_list))



    transforms.append(lcp_T.ToDict())

    if network_function is not None:
        transforms += [
            ComputeSpatial(network_function()),
        ]

    transforms = T.Compose(transforms)

    return transforms
