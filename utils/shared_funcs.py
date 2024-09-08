#Shared functions between the different networks
import logging
import math
import os
from functools import partial

import numpy as np
import scipy
import torch
import torch.nn.functional as F
import torchsparse.nn as spnn
import yaml
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
from torch import nn

import networks
import utils.metrics as metrics
from lightconvpoint.datasets.dataset import get_dataset
from utils.collate_function import collate_function
from utils.logging_files_functions import train_log_data_da, val_log_data_da
from utils.transforms import da_get_inputs, da_get_transforms


def da_sf_get_dataloader(target_DatasetClass, config, net, network_function, val=1, train_shuffle=True, cat_list_changed=[], drop_last=False, keep_orignal_data=False):
    target_train_transforms = da_get_transforms(config, network_function, train=True, source_flag=False, keep_orignal_data=keep_orignal_data)
    target_test_transforms = da_get_transforms(config, network_function, train=False, source_flag=False)

    target_train_dataset = target_DatasetClass(config["target_dataset_root"], 
               split=config["train_split"], 
               transform=target_train_transforms,
               da_flag = True,config=config, domain="target")

    # build the test datasets for target and source
    if val == 1: 
        target_test_dataset = target_DatasetClass(config["target_dataset_root"],
                split=config["val_split"], 
                transform=target_test_transforms,
                da_flag = True, config=config, domain="target")

    
    # create the collate function
    if len(cat_list_changed)>0:
        cat_item_list = cat_list_changed + net.get_cat_item_list()
    else: 
        cat_item_list = ["pos", "x", "y", 'shape_id', 'N'] + net.get_cat_item_list()
        

    stack_item_list = [] + net.get_stack_item_list()
    if keep_orignal_data:
        sparse_item_list_train = ["sparse_input", "original_sparse_input"]
    else:
        sparse_item_list_train = ["sparse_input"]
    sparse_item_list_test = ["sparse_input"]
    
    collate_fn_train = partial(collate_function, 
                cat_item_list = cat_item_list,
                stack_item_list = stack_item_list,
                sparse_item_list = sparse_item_list_train
            )
    
    collate_fn_test = partial(collate_function, 
                cat_item_list = cat_item_list,
                stack_item_list = stack_item_list,
                sparse_item_list = sparse_item_list_test
            )

    #build the data loaders for TARGET
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset,
        batch_size=config["training_batch_size"],
        shuffle=train_shuffle,
        pin_memory=True,
        num_workers=config["threads"],
        collate_fn=collate_fn_train, drop_last=drop_last
       )
    
    target_test_loader = torch.utils.data.DataLoader(
       target_test_dataset,
       batch_size=config["test_batch_size"],
       shuffle=False,
       pin_memory=True,
       num_workers=config["threads"],
       collate_fn=collate_fn_test, drop_last=drop_last
    )
    return {"target_train_loader": target_train_loader, "target_test_loader": target_test_loader}




def optimizer_selection(logging, config, net, net_parameters=None):
        if net_parameters is None: 
            #Train the complete network, otherwise only the parameters in net_parameters are trained
            net_parameters = net.parameters()
        

        if config["optimizer"] == "AdamW":
            logging.info(f"Selected optimizer: {config['optimizer']}, LR(start): {config['training_lr_start']}")
            optimizer = torch.optim.AdamW(net_parameters,config["training_lr_start"])
        elif config["optimizer"] == "Adam":
            betas = (0.9, 0.99)
            logging.info(f"Selected optimizer: {config['optimizer']}, LR(start): {config['training_lr_start']}, betas: {betas}")
            optimizer = torch.optim.Adam(net_parameters, lr=config["training_lr_start"], betas=betas)
        else: 
            raise NotImplementedError

        return optimizer

def learning_rate_scheduler_selection(logging, config, optimizer):
        if config['lr_scheduler'] is None:
            ## Learning rate is kept constant
            logging.info("No LR rate scheduling")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200000, gamma=1.0)
        elif config['lr_scheduler'] == "step_lr":
            #Every step the learning rate is multiplied with gamma 
            logging.info(f"Step LR rate scheduling with step size: {config['step_lr_step_size']}, gamma: {config['step_lr_gamma']}")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_lr_step_size'], gamma=config['step_lr_gamma'])
        elif config['lr_scheduler'] == "cos_an_lr":
            #Cosine annealing
            T_max = 10000
            eta_min = 0
            logging.info(f"Cosine annealing LR rate scheduling with T_max: {T_max}, eta_min: {eta_min}")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        elif config['lr_scheduler'] == "cos_an_half_lr":
            #Cosine annealing
            T_max = config['training_iter_nbr']
            eta_min = 0
            logging.info(f"Cosine annealing LR rate scheduling with only half period T_max: {T_max}, eta_min: {eta_min}")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        else: 
            raise NotImplementedError

        return scheduler

def deactivation_bn(net, config, logging):
    number_batchnorm_parameter = 0
    for name, layer in net.named_parameters():
        #breakpoint()
        if "projection" in name or "dual_seg_head" in name or "classifier" in name or name.split(".")[-1] == "kernel": 
            pass
        else:
            
            if name.split(".")[1] == "point_transforms" and name.split(".")[-2] == '0' :
                pass
            else:
                #print("Name {}".format(name))
                number_batchnorm_parameter=number_batchnorm_parameter+1
                layer.requires_grad=config["bn_grad"] if "bn_grad" in config else True
    logging.info("Total numbers of deactivated batchnorm parameter layers {}".format(number_batchnorm_parameter))
    #assert number_batchnorm_parameter == 104

    logging.info("Setting the BN momentum to {}, BN set to eval {}".format(config["bn_momentum"], config["bn_run_stats_freeze"]))
    for m in net.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            if  config["bn_run_stats_freeze"]: 
                m.eval()
            else:
                m.train()
            m.momentum = config["bn_momentum"]
    return 


def construct_network(config, logging):
        latent_size = config["network_latent_size"]
        backbone = config["network_backbone"]
        decoder = {'name':config["network_decoder"], 'k': config['network_decoder_k']}
        logging.info("Backbone {}".format(backbone))
        logging.info("Decoder {}".format(decoder))
        in_channels_source, _, in_channels_target, _ = da_get_inputs(config)
        logging.info("In channels source: {}".format(in_channels_source))
        logging.info("in channels target: {}".format(in_channels_target))
        logging.info("Creating the network:")
        def network_function():
            return networks.Network(in_channels_source, latent_size, config["network_n_labels"], backbone, decoder, 
                        voxel_size=config["voxel_size"],
                        cylindrical_coordinates=config["cylindrical_coordinates"], \
                        dual_seg_head = config["dual_seg_head"],
                        nb_classes = config["nb_classes"],
                        da_flag=False, target_in_channels=in_channels_target, config=config)
        net = network_function()
        return net, network_function


def calculation_metrics(metrics_holder, outputs, occupancies, loss_seg, loss, recons_loss, output_seg=None, source_data=None, ignore_list=[], output_data=None):
    output_np = (torch.sigmoid(outputs).cpu().detach().numpy() > 0.5).astype(int)
    target_np = occupancies.cpu().numpy().astype(int)
    
    metrics_holder.counter +=1
    #### Semantic Segmentation
    output_seg_np = np.argmax(output_seg[:,1:].cpu().detach().numpy(), axis=1) + 1 #As the 0 dimension (class ignore) should not be taken into account for finding the max value
    target_seg_np = source_data["y"].cpu().numpy().astype(int)

    softmaxed_value = scipy.special.softmax(output_seg.cpu().detach().numpy(), axis=1)
    entropy_calculated = entropy(softmaxed_value, axis=1)
    entropy_mean = np.average(entropy_calculated, axis=0)
    entropy_on_average_prob = entropy(np.average(softmaxed_value, axis=0))
    information_maximation = entropy_on_average_prob - entropy_mean
    metrics_holder.entropy_average_sum += entropy_on_average_prob
    metrics_holder.information_maximation_sum +=information_maximation

    entropy_average_value = metrics_holder.entropy_average_sum/metrics_holder.counter
    information_maximation_value = metrics_holder.information_maximation_sum/metrics_holder.counter
    ##Batch nuclear norm is not caluclated during training to save time
    bnm_value_avg = 0.0

    cm_seg_head_ = confusion_matrix(target_seg_np.ravel(), output_seg_np.ravel(), labels=list(range(metrics_holder.config["nb_classes"])))
    metrics_holder.cm_seg_head += cm_seg_head_
    train_seg_head_oa = metrics.stats_overall_accuracy(metrics_holder.cm_seg_head, ignore_list=ignore_list)
    train_seg_head_maa, train_acc_per_class = metrics.stats_accuracy_per_class(metrics_holder.cm_seg_head, ignore_list=ignore_list) #First return value is the mean IoU
    train_seg_head_miou, train_seg_iou_per_class = metrics.stats_iou_per_class(metrics_holder.cm_seg_head, ignore_list=ignore_list) #First return value is the mean IoU
    if metrics_holder.config["dual_seg_head"] and not metrics_holder.target:
        metrics_holder.error_seg_head += loss_seg.item()
        # point wise scores on training segmentation head
        train_seg_head_loss = metrics_holder.error_seg_head / metrics_holder.cm_seg_head.sum()
    
    else:
        train_seg_head_loss=0

    #### Occupany
    cm_ = confusion_matrix(
        target_np.ravel(), output_np.ravel(), labels=list(range(metrics_holder.cm.shape[0]))
    )
    metrics_holder.cm += cm_

    metrics_holder.error += loss.item()
    metrics_holder.error_recons += recons_loss.item()

    # point wise scores on training
    train_oa = metrics.stats_overall_accuracy(metrics_holder.cm)
    train_aa = metrics.stats_accuracy_per_class(metrics_holder.cm)[0]
    train_iou = metrics.stats_iou_per_class(metrics_holder.cm)[0]
    train_aloss = metrics_holder.error / metrics_holder.cm.sum()
    train_aloss_recons = metrics_holder.error_recons / metrics_holder.cm.sum()
    train_aloss_additional = metrics_holder.error_additional / metrics_holder.cm.sum()

    ### Semantic Segmentation for Occupancy
    if metrics_holder.config["in_seg_loss"]:
        pass
    
    if metrics_holder.config["dual_seg_head"]:
        return_data =  {"train_oa":train_oa, "train_aa":train_aa, "train_aloss":train_aloss,"train_aloss_recons":train_aloss_recons, "train_aloss_additional":train_aloss_additional,\
             "train_iou":train_iou, "train_seg_head_miou":train_seg_head_miou,"train_seg_head_maa":train_seg_head_maa, "train_seg_head_loss": train_seg_head_loss,\
            "accuracy_per_class": train_acc_per_class, "seg_iou_per_class": train_seg_iou_per_class, "entropy_mean_avg":entropy_average_value[0], "information_maximization_avg":information_maximation_value[0], "bnm_value_avg":bnm_value_avg}
        return return_data
    else:
        return  {"train_oa":train_oa, "train_aa":train_aa, "train_iou": train_iou, "train_aloss": train_aloss, "train_aloss_recons": train_aloss_recons, "train_aloss_additional": train_aloss_additional}


def get_savedir_root(config, experiment_name):
    
    savedir = f"{experiment_name}_{config['network_backbone']}_{config['network_decoder']}_{config['network_decoder_k']}"
    savedir += f"_{config['train_split']}Split"
    if ("desc" in config) and config["desc"]:
        savedir += f"_{config['desc']}"

    savedir_root = os.path.join(config["save_dir"], savedir)

    return savedir_root

def get_bbdir_root(config):
    # Gives the backbone dir 
    savedir_root = config["da_fixed_head_path_model"]

    return savedir_root

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_config_file(config, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def ignore_selection(idx=-1):
    if idx == -1: 
        return []
    elif idx == 0: 
        return [0]
    elif idx == 1: 
        #SK original classes (20), mapped to noise of DA
        return [0, 7, 8, 12, 13, 14, 18, 19]
    elif idx == 2: 
        #NS original classes (17), mapped to noise of NS 
        return [0, 1, 8, 12, 15]
    else: 
        return [] 

def minent_entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        input: number_points x channels x 1
        output: batch_size x 1 x h x w
        Taken and adapted from: https://github.com/valeoai/ADVENT/blob/master/advent/utils/loss.py
    """
    
    assert v.dim() == 3 or v.dim() == 2
    if  v.dim() == 3:
        v = F.softmax(v, dim=1)
        n, c, h = v.size()
        loss = -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * np.log2(c))
    else:
        v = F.softmax(v, dim=1)
        n, c = v.size()
        loss = -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * np.log2(c))
    return loss


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics

    for m in model.modules():
        if isinstance(m, spnn.BatchNorm) or isinstance(m, nn.BatchNorm1d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model