import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import utils.metrics as metrics
from lightconvpoint.utils.misc import dict_to_device


def class_prior_class_names(config, logging):
    logging.info(f"Using the source dataset class distribution as priors")
    if (config["source_dataset_name"] == "NuScenes" and config["target_dataset_name"] == "SemanticKITTI"):
        class_prior = np.array([0.05998, 0.00023, 0.00065, 0.02492, 0.01812, 0.00358, 0.48945, 0.10799, 0.10631, 0.18878]).astype(np.float32)
        names_list=["ignore", "car", "bicycle", "motorcycle", "truck", "other_vehicle", "pedestrian", "driveable_surface", "sidewalk", "terrain", "vegetation"]
    
    elif config["target_dataset_name"] == "NuScenes" and config["source_dataset_name"] == "SemanticKITTI":
        class_prior = np.array([0.0559719,	0.0002205,	0.0005269,	0.0025816,	0.0030712,	0.0004598,	0.2823960,	0.1903702,	0.1033538,	0.3610481]).astype(np.float32)
        names_list=["ignore", "car", "bicycle", "motorcycle", "truck", "other_vehicle", "pedestrian", "driveable_surface", "sidewalk", "terrain", "vegetation"]
        

    elif config["source_dataset_name"] == "SynLidar" and config["target_dataset_name"] == "SemanticKITTI": 
        class_prior = np.array([0.0148227,	0.0009359,	0.0024249,	0.0197668,	0.0146102,	0.0065604,	0.0050679,\
                                0.0079493,	0.4276948,	0.0024630,	0.1321319,	0.0054249,	0.1546558,	0.0243836,\
                                0.0703529,	0.0076416,	0.0904148,	0.0103850,	0.0023139]).astype(np.float32)
        names_list = ["ignore","car","bicycle","motorcycle","truck","other_vehicle","pedestrian","bicyclist",\
    "motorcyclist","road","parking","sidewalk","other_ground","building","fence","vegetation","trunk","terrain","pole","traffic_sign"]
        
    elif config["source_dataset_name"] == "SynLidar" and config["target_dataset_name"] == "SemanticPOSS":
        class_prior = np.array([0.00888823,	0.01763608,	0.0200823,	0.01035303,	0.09531635,	0.00313491,	0.01406994,\
                                0.00654064,	0.20953264,	0.00068746,	0.03303564,	0.00126801,	0.57945477]).astype(np.float32)
        names_list = ["ignore", "person", "rider", "car", "trunk", "plants", "traffic_sign", "pole",\
                    "garbage_can", "building", "cone", "fence", "bike", "ground"]
    
    elif config["source_dataset_name"] == "NuScenes" and (config["target_dataset_name"] == "NuScenes"):
        class_prior = np.array([0.0143828,	0.0001923,	0.0055170,	0.0652271,	0.0023115,	0.0001922,	0.0028795,\
                                0.0013829,	0.0123440,	0.0302873,	0.4208411,	0.0123001,	0.0896406,	0.0263130,\
                                0.2382568,	0.0779318]).astype(np.float32)
        names_list=["ignore", "barrier", "bicycle", "bus", "car", \
                "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck", \
                "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation"]
    elif config["source_dataset_name"] == "NuScenes" and config["target_dataset_name"] == "SemanticPOSS":
        class_prior = np.array([0.00273356,	0.00067204,	0.07901461,	0.54972416,	0.14491832,	0.22293731]).astype(np.float32)
        names_list=["ignore", "person", "bike",	"car","ground","vegetation","manmade"]
    
    elif config["source_dataset_name"] == "NuScenes" and config["target_dataset_name"] == "Pandaset":
        class_prior = np.array([0.00067204,	0.00273356,	0.3757292,	0.0829,	0.09109496,	0.22293731,	0.14491832,	0.07901461]).astype(np.float32)
        names_list=["ignore", "wheeled2","pedestrian","driveable_ground", "sidewalk", "other_ground", "manmade", "vegetation", "wheeled4"]
    elif config["source_dataset_name"] == "NuScenes" and config["target_dataset_name"] == "Waymo":
        class_prior = np.array([0.059987716,0.000225618,0.000649914,0.024919322,0.018033373,0.003561290,\
            0.489500864,0.108002309,0.106319672,0.188799922]).astype(np.float32)
        names_list=["ignore","Car","Bicycle","Motorcycle","Truck","Other_Vehicle","Pedestrian","Driveable_surface","Sidewalk",\
            "Walkable","Vegetation"]
    else: 
        raise Exception(f"Class distribution not available for: source {config['source_dataset_name']} target {config['target_dataset_name']} ")

    
    return class_prior, names_list

def source_model_source_class_confusion_matrix(config, logging):
    base_path = '/home/bmichele/workspace/DASF/ckpts_bn/'
    final_path = None
    
    if not config["parameter"]["prior_target"]:
        logging.info(f"Using the source dataset class distribution as priors")
        if (config["source_dataset_name"] == "NuScenes" and config["target_dataset_name"] == "SemanticKITTI"):
            final_path = 'rep0_EVAL_ns_sk_no_premap/last_ckpt_validation_cm_source_w_ignore.csv'

        elif config["source_dataset_name"] == "SynLidar" and config["target_dataset_name"] == "SemanticKITTI": 
            final_path = 'rep0_EVAL_syn_sk_no_premap/last_ckpt_validation_cm_source_w_ignore.csv'

        elif config["source_dataset_name"] == "SynLidar" and config["target_dataset_name"] == "SemanticPOSS":
            final_path = 'rep0_EVAL_syn_poss_no_premap/last_ckpt_validation_cm_source_w_ignore.csv'

            
        elif config["source_dataset_name"] == "NuScenes" and config["target_dataset_name"] == "SemanticPOSS":
            final_path = 'rep0_EVAL_ns_poss_no_premap/last_ckpt_validation_cm_source_w_ignore.csv'
        else: 
            raise ValueError(f"Class distribution not available for: source {config['source_dataset_name']} target {config['target_dataset_name']} ")
    else:
        logging.info(f"Using the source dataset class distribution as priors")
        if (config["source_dataset_name"] == "NuScenes" and config["target_dataset_name"] == "SemanticKITTI"):
            final_path = 'rep0_EVAL_ns_sk_no_premap/last_ckpt_validation_cm_target_w_ignore.csv'

        elif config["source_dataset_name"] == "SynLidar" and config["target_dataset_name"] == "SemanticKITTI": 
            final_path = 'rep0_EVAL_syn_sk_no_premap/last_ckpt_validation_cm_target_w_ignore.csv'

        elif config["source_dataset_name"] == "SynLidar" and config["target_dataset_name"] == "SemanticPOSS":
            final_path = 'rep0_EVAL_syn_poss_no_premap/last_ckpt_validation_cm_target_w_ignore.csv'

            
        elif config["source_dataset_name"] == "NuScenes" and config["target_dataset_name"] == "SemanticPOSS":
            final_path = 'rep0_EVAL_ns_poss_no_premap/last_ckpt_validation_cm_target_w_ignore.csv'
        else: 
            raise ValueError(f"Class distribution not available for: source {config['source_dataset_name']} target {config['target_dataset_name']} ")
    
    df = pd.read_csv(os.path.join(base_path, final_path),delimiter=' ', header=None)
    target_confusion_matrix = df.to_numpy()
    normalized_per_row = target_confusion_matrix/(target_confusion_matrix.sum(axis=1)[:,None])
    normalized_per_row[np.isnan(normalized_per_row)] = 0  #Set rows to zero, when e.g. they do not appear in the test-set
    cost_matrix = 1 - normalized_per_row 
    np.fill_diagonal(cost_matrix, 0)
    #Not using the ignore classes
    cost_matrix = cost_matrix[1:,1:]
    return cost_matrix



def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_model(model, use_running_mean=False):
    """_summary_

    Args:
        model (_type_): _description_
        use_running_mean (bool, optional): Flag if the running means should be used (in the original paper they use the instance norm) . Defaults to False.

    Returns:
        _type_: Model with adapted BN layers 
    """
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.requires_grad_(True)

            if not use_running_mean:
                # Tent default
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            else: 
                #Use the running statistics from the training
                m.eval()
    return model


def configure_freeze_models(model, config, list_parameter_to_update, list_parameter_others):
    """
    Freeze and unfreeze certain parts of the model, based on the settings in config.
    Args:
        model (_type_): Model which layer should be froozen
        config (Dict): Config dictionary
        list_parameter_to_update (List): List of parameters that get updated
        list_parameter_others (List): List of parameters that do not get updated
    Returns:
        model: adapted model
        list_parameter_to_update (List): List of parameters that get updated
        list_parameter_others (List): List of parameters that do not get updated
    """
    for name, parameter in model.named_parameters():
        if not config["parameter"]["finetune"]: 
            if name == "net.ae_0_learned" or name == "net.be_0_learned" or name == "net.ae_1_learned" or name == "net.be_1_learned" or name == "net.parametrizations.ae_0_learned.original" or name == "net.parametrizations.ae_1_learned.original" or "ae_learned" in name or "be_learned" in name:
                parameter.requires_grad=True
                list_parameter_to_update.append(parameter)
                
            else: 
                parameter.requires_grad=False
                list_parameter_others.append(parameter)
        else:
            if config["parameter"]["fintune_setting"] == "complete_finetune":
                parameter.requires_grad=True
                list_parameter_to_update.append(parameter)
                
            elif config["parameter"]["fintune_setting"] == "shot_finetune":
                if "dual_seg_head.weight" == name or "dual_seg_head.bias" == name: 
                    parameter.requires_grad=False
                    list_parameter_others.append(parameter)
                else: 
                    parameter.requires_grad=True
                    list_parameter_to_update.append(parameter)
                    
            elif config["parameter"]["fintune_setting"] == "ll_and_scalable_finetune": 
                if name == "net.ae_0_learned" or name == "net.be_0_learned" or name == "net.ae_1_learned" or name == "net.be_1_learned" or name == "net.parametrizations.ae_0_learned.original" or name == "net.parametrizations.ae_1_learned.original" or "ae_learned" in name or "be_learned" in name:
                    parameter.requires_grad=True
                    list_parameter_to_update.append(parameter)
                    
                elif "dual_seg_head.weight" == name or "dual_seg_head.bias" == name:
                    parameter.requires_grad=True
                    list_parameter_to_update.append(parameter)
                   
                else: 
                    parameter.requires_grad=False
                    list_parameter_others.append(parameter)
            elif config["parameter"]["fintune_setting"] == "classic": 
                if "dual_seg_head.weight" == name or "dual_seg_head.bias" == name:
                    parameter.requires_grad=True
                    list_parameter_to_update.append(parameter)
                   
                
                else: 
                    parameter.requires_grad=True
                    list_parameter_others.append(parameter)
                    
             
            elif  config["parameter"]["fintune_setting"] == "LL":
                #In the case of finetuning only the classifier layer is adapted
                if "dual_seg_head.weight" == name or "dual_seg_head.bias" == name: 
                    parameter.requires_grad=True
                    list_parameter_to_update.append(parameter)
                    
                else: 
                    parameter.requires_grad=False
                    list_parameter_others.append(parameter)
            else: 
                raise ValueError(f"Unknown finetune setting: {config['parameter']['fintune_setting']}")
    return model, list_parameter_to_update, list_parameter_others