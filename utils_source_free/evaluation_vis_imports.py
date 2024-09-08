import os

from matplotlib.pyplot import fill
import numpy as np
import yaml
from tqdm import tqdm
import logging
import shutil
import pickle
from sklearn.metrics import confusion_matrix
import torch_geometric.transforms as T
import pdb

# torch imports
import torch
import torch.nn.functional as F

# lightconvpoint imports
from lightconvpoint.datasets.dataset import get_dataset
import lightconvpoint.utils.transforms as lcp_T
from lightconvpoint.utils.logs import logs_file
from lightconvpoint.utils.misc import dict_to_device

import utils.argparseFromFile as argparse
from utils.utils import wblue, wgreen
import utils.metrics as metrics

from utils.transforms import da_get_inputs, da_get_transforms
from utils.utils import get_savedir_root, get_bbdir_root
from utils.logging_files_functions import val_log_data_da, train_log_data_da
from utils.shared_funcs import get_savedir_root, get_bbdir_root, collate_function, count_parameters, save_config_file, metrics_holder, ignore_selection, da_sf_get_dataloader 

from utils.transforms import CreateDirs, CreateInputs, CreateNonManifoldPoints, UseAsFeatures, Scaling, ComputeSpatial, Transpose, VoxelDecimation, Dupplicate
import datasets
import networks
import pandas as pd

from scipy.spatial import KDTree

from datetime import datetime
from functools import partial

from config import ex

from torch.utils.tensorboard import SummaryWriter
from configs_and_mappings import name_shift, read_yaml_file #config,

from utils_source_free.learn_mapping_utils import class_prior_class_names, configure_model
from utils_source_free.utils import summation_matrix_generator, sf_class_mapping_loader, validation_non_premap, validation_non_premap_instancebn
from utils_source_free.utils import prediction_changer

config={
    "fast_rep_flag": False,
    # Dataset configs
    "dataset_name" : 'UnknwonDataset',
    "dataset_root" : 'UnknownDatasetPath',
    "source_dataset_name" : 'UnknwonDataset',
    "source_dataset_root" : 'UnknownDatasetPath',
    "target_dataset_name" : 'UnknwonDataset',
    "target_dataset_root" : 'UnknownDatasetPath',
    "save_dir" : 'results_ckpt',
    "ns_dataset_version":'v1.0-trainval',
    #"ns_dataset_version":'v1.0-mini',

    # splits
    "train_split" : 'train',
    "val_split" : 'val',
    "test_split" : 'val',
    "nb_classes" : 1,

    # Method parameters
    "input_intensities" : False,
    "input_dirs":False,
    "input_normals":False,
    "source_input_intensities":False,
    "source_input_dirs": False,    
    "target_input_intensities":False,
    "target_input_dirs" :False,

    "manifold_points" : 10000,
    "non_manifold_points" :2048,
    
    # Training parameters
    "da_flag": False,
    "dual_seg_head" : True,
    "training_iter_nbr" : 50000,
    "training_batch_size" : 1,
    "test_batch_size" : 1,
    "training_lr_start" : 0.001,
    "training_lr_start_head" : None,
    "optimizer" : "AdamW",
    "lr_scheduler" : None,
    "step_lr_step_size" :200000, #Step size for steplr scheduler
    "step_lr_gamma" : 0.7, #Gamma for steplr scheduler
    "voxel_size" : 0.1,
    "val_interval" :5,
    "resume" :False,

    # Network parameter
    "network_backbone" : 'TorchSparseMinkUNet', #_normalised',
    "network_latent_size": 128,
    "network_decoder" : 'InterpAllRadiusNet',
    "network_decoder_k" : 2.0,
    "network_n_labels" : 1,
    "use_no_dirs_rec_head_flag" : False,
    "rotation_head_flag" : False,

    # Technical parameter
    "device" : 'cuda',
    "threads" :6,
    "interactive_log" : True,
    "logging" : 'INFO',
    "use_amp" : False,

    # Data augmentation
    "randRotationZ" : True,
    "randFlip" : True,
    "no_augmentation" : False,
    
    # Ckpt path 
    "ckpt_path_model" : "UnknownPath",

    # Weighting parameter for loss
    "weight_rec_src" : 1.0,
    "weight_rec_trg" : 1.0,
    "weight_ss_src" : 1.0,
    "weight_ss_trg" : 1.0,
    "weight_inside_seg_src" : 1.0,

    # Ignorance idx
    "ignore_idx" : 0,
    "get_latent" : False,


    # Test flag
    "test_flag_eval" : False,
    "target_training" : True,
    "source_training" : True,

    # Which ckpt to load from in eval
    "ckpt_number" : -1,
    "cosmix_backbone" : False,
    
    "da_flag" : True,
    "dual_seg_head" : True,
    "source_dataset_name" : 'NuScenes',
    "source_dataset_root" : 'data/nuscenes',
    #"source_dataset_name" : 'SemanticKITTI',
    #"source_dataset_root" : 'data/SemanticKITTI',
    
    "source_dataset_name" : 'SynLidar',
    "source_dataset_root" : 'data/synlidar',
    "source_input_intensities" : False,
    "source_input_dirs" : False,
    #"nb_classes" : 11,
    "nb_classes" : 20,
    #"target_dataset_name" : 'NuScenes',
    #"target_dataset_root" : 'data/nuscenes',
    "target_dataset_name" : 'SemanticKITTI',
    "target_dataset_root" : 'data/SemanticKITTI',
    #"target_dataset_name" : 'SemanticPOSS',
    #"target_dataset_root" : 'data/SemanticPOSS',
    "target_input_intensities" : False,
    "target_input_dirs" : False,
    "lr_scheduler":"step_lr",
    "training_iter_nbr":200000,
    "step_lr_gamma":0.7,
    
    #Legacy parameter
    "cylindrical_coordinates":False,
    "adv_flag":False,
    "dua_da_flag":False,
    "planar_manifold_flag":False,
    "planar_interpolation_manifold_flag":False,
    "sampling_manifold":False, 
    "rendered_input":False, 
    "in_seg_loss":False, 
    "intensity_loss":False
}


def write_non_premap_3(list_write,save_path, alpha, mapping, net_so, net_best, net_last, mapped, config, test_loader, N_LABELS, epoch, disable_log, device, ce_loss_layer, loss_layer, target_flag=False, list_ignore_classes=[], eval_flag=True):
    os.makedirs(save_path, exist_ok=True)
    
    if eval_flag:
        net_so.eval()
        net_best.eval()
        net_last.eval()

    
    
    set_write = set(list_write)
    mapping_information = sf_class_mapping_loader(source_dataset=config["source_dataset_name"],target_dataset=config["target_dataset_name"])
    with torch.no_grad():
        count_iter = 0
        t = tqdm(test_loader,desc="  Test " + str(epoch),ncols=200,disable=disable_log,)
        for data in t:
            if count_iter in set_write: 
                data = dict_to_device(data, device)

                target_seg_np = data["y"].cpu().numpy().astype(int)
                pos = data["pos"].cpu().numpy()

                #Source only 
                _, output_seg_so, _ = net_so.forward_mapped_learned(data, mapped, alpha) #SOURCE data
                output_seg_np_so = prediction_changer(output_seg_so.cpu().detach(), mapping_information)
                cm_seg_head_so = confusion_matrix(target_seg_np.ravel(), output_seg_np_so.ravel(), labels=list(range(config["nb_classes_inference"])))
                test_seg_head_maa_so, accuracy_per_class_so = metrics.stats_accuracy_per_class(cm_seg_head_so, ignore_list=list_ignore_classes) #First return value is the mean IoU
                test_seg_head_miou_so, seg_iou_per_class_so = metrics.stats_iou_per_class(cm_seg_head_so, ignore_list=list_ignore_classes) #First return value is the mean IoU



                #Best choice
                _, output_seg_best, _ = net_best.forward_mapped_learned(data, mapped, alpha) #SOURCE data
                output_seg_np_best = prediction_changer(output_seg_best.cpu().detach(), mapping_information)
                cm_seg_head_best = confusion_matrix(target_seg_np.ravel(), output_seg_np_best.ravel(), labels=list(range(config["nb_classes_inference"])))
                test_seg_head_maa_best, accuracy_per_class_best = metrics.stats_accuracy_per_class(cm_seg_head_best, ignore_list=list_ignore_classes) #First return value is the mean IoU
                test_seg_head_miou_best, seg_iou_per_class_best = metrics.stats_iou_per_class(cm_seg_head_best, ignore_list=list_ignore_classes) #First return value is the mean IoU




                #20K 
                _, output_seg_last, _ = net_last.forward_mapped_learned(data, mapped, alpha) #SOURCE data
                output_seg_np_last = prediction_changer(output_seg_last.cpu().detach(), mapping_information)
                cm_seg_head_last = confusion_matrix(target_seg_np.ravel(), output_seg_np_last.ravel(), labels=list(range(config["nb_classes_inference"])))
                test_seg_head_maa_last, accuracy_per_class_last = metrics.stats_accuracy_per_class(cm_seg_head_last, ignore_list=list_ignore_classes) #First return value is the mean IoU
                test_seg_head_miou_last, seg_iou_per_class_last = metrics.stats_iou_per_class(cm_seg_head_last, ignore_list=list_ignore_classes) #First return value is the mean IoU


                merged_pc = np.hstack((pos[:,1:], target_seg_np[:,None], output_seg_np_so,output_seg_np_best, output_seg_np_last))

                name_pc = f"ns_sk_{count_iter}.xyz"

                np.savetxt(os.path.join(save_path, name_pc), merged_pc)
                print(merged_pc.shape)

            count_iter += 1
            if count_iter % 10 == 0:
                torch.cuda.empty_cache()
            
        return 
    

def validation_non_premap_3(alpha, mapping, net_so, net_best, net_last, mapped, config, test_loader, N_LABELS, epoch, disable_log, device, ce_loss_layer, loss_layer, target_flag=False, list_ignore_classes=[], eval_flag=True):
    if eval_flag:
        net_so.eval()
        net_best.eval()
        net_last.eval()

    list_net_so_miou = []
    list_net_so_maa = []
    list_net_so_class_iou = None
    list_net_so_class_aa = None
    
    
    list_net_best_miou = []
    list_net_best_maa = []
    list_net_best_class_iou = None
    list_net_best_class_aa = None
    
    list_net_last_miou = []
    list_net_last_maa = []
    list_net_last_class_iou = None
    list_net_last_class_aa = None
    
    
    
    mapping_information = sf_class_mapping_loader(source_dataset=config["source_dataset_name"], target_dataset=config["target_dataset_name"])
    with torch.no_grad():
        count_iter = 0
        t = tqdm(test_loader,desc="  Test " + str(epoch),ncols=200,disable=disable_log,)
        for data in t:

            data = dict_to_device(data, device)
            
            target_seg_np = data["y"].cpu().numpy().astype(int)
            
            #Source only 
            _, output_seg_so, _ = net_so.forward_mapped_learned(data, mapped, alpha) #SOURCE data
            output_seg_np_so = prediction_changer(output_seg_so.cpu().detach(), mapping_information)
            cm_seg_head_so = confusion_matrix(target_seg_np.ravel(), output_seg_np_so.ravel(), labels=list(range(config["nb_classes_inference"])))
            test_seg_head_maa_so, accuracy_per_class_so = metrics.stats_accuracy_per_class(cm_seg_head_so, ignore_list=list_ignore_classes) #First return value is the mean IoU
            test_seg_head_miou_so, seg_iou_per_class_so = metrics.stats_iou_per_class(cm_seg_head_so, ignore_list=list_ignore_classes) #First return value is the mean IoU
            
            
            list_net_so_miou.append(test_seg_head_miou_so)
            list_net_so_maa.append(test_seg_head_maa_so)
            
            if list_net_so_class_iou is None: 
                list_net_so_class_iou = seg_iou_per_class_so
                list_net_so_class_aa = accuracy_per_class_so
            else: 
                list_net_so_class_iou = np.vstack((list_net_so_class_iou,seg_iou_per_class_so))
                list_net_so_class_aa = np.vstack((list_net_so_class_aa,accuracy_per_class_so))
            
            
            del output_seg_np_so
            del cm_seg_head_so
            
            #Best choice
            _, output_seg_best, _ = net_best.forward_mapped_learned(data, mapped, alpha) #SOURCE data
            output_seg_np_best = prediction_changer(output_seg_best.cpu().detach(), mapping_information)
            cm_seg_head_best = confusion_matrix(target_seg_np.ravel(), output_seg_np_best.ravel(), labels=list(range(config["nb_classes_inference"])))
            test_seg_head_maa_best, accuracy_per_class_best = metrics.stats_accuracy_per_class(cm_seg_head_best, ignore_list=list_ignore_classes) #First return value is the mean IoU
            test_seg_head_miou_best, seg_iou_per_class_best = metrics.stats_iou_per_class(cm_seg_head_best, ignore_list=list_ignore_classes) #First return value is the mean IoU
            
            
            
            list_net_best_miou.append(test_seg_head_miou_best)
            list_net_best_maa.append(test_seg_head_maa_best)
            
            if list_net_best_class_iou is None: 
                list_net_best_class_iou = seg_iou_per_class_best
                list_net_best_class_aa = accuracy_per_class_best
            else: 
                list_net_best_class_iou = np.vstack((list_net_best_class_iou, seg_iou_per_class_best))
                list_net_best_class_aa = np.vstack((list_net_best_class_aa, accuracy_per_class_best))
            
            del output_seg_np_best
            del cm_seg_head_best
            
            #20K 
            _, output_seg_last, _ = net_last.forward_mapped_learned(data, mapped, alpha) #SOURCE data
            output_seg_np_last = prediction_changer(output_seg_last.cpu().detach(), mapping_information)
            cm_seg_head_last = confusion_matrix(target_seg_np.ravel(), output_seg_np_last.ravel(), labels=list(range(config["nb_classes_inference"])))
            test_seg_head_maa_last, accuracy_per_class_last = metrics.stats_accuracy_per_class(cm_seg_head_last, ignore_list=list_ignore_classes) #First return value is the mean IoU
            test_seg_head_miou_last, seg_iou_per_class_last = metrics.stats_iou_per_class(cm_seg_head_last, ignore_list=list_ignore_classes) #First return value is the mean IoU
            
            list_net_last_miou.append(test_seg_head_miou_last)
            list_net_last_maa.append(test_seg_head_maa_last)
            
            if list_net_last_class_iou is None: 
                list_net_last_class_iou = seg_iou_per_class_last
                list_net_last_class_aa = accuracy_per_class_last
            else: 
                list_net_last_class_iou = np.vstack((list_net_last_class_iou,seg_iou_per_class_last))
                list_net_last_class_aa = np.vstack((list_net_last_class_aa,accuracy_per_class_last))
            
            del output_seg_np_last
            del cm_seg_head_last
                        
            count_iter += 1
            if count_iter % 10 == 0:
                torch.cuda.empty_cache()
            
        return {'list_net_so_miou':list_net_so_miou,
        'list_net_so_maa':list_net_so_maa,
        'list_net_so_class_iou':list_net_so_class_iou,
        'list_net_so_class_aa':list_net_so_class_aa,
        'list_net_best_miou':list_net_best_miou,
        'list_net_best_maa':list_net_best_maa,
        'list_net_best_class_iou':list_net_best_class_iou,
        'list_net_best_class_aa':list_net_best_class_aa,
        'list_net_last_miou':list_net_last_miou,
        'list_net_last_maa':list_net_last_maa,
        'list_net_last_class_iou':list_net_last_class_iou,
        'list_net_last_class_aa':list_net_last_class_aa}