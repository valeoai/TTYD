
import torch
import numpy as np
from tqdm import tqdm
from configs_and_mappings import read_yaml_file #config,

import os
import numpy as np
import torch
from tqdm import tqdm
from lightconvpoint.utils.misc import dict_to_device
import utils.metrics as metrics
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
import scipy

def config_adapter(config):
    if config["parameter"]["setting"] == "NS2SK":
        config["source_dataset_name"] = 'NuScenes'
        config["source_dataset_root"] = 'data/nuscenes'
        config["nb_classes"] = 17
        config["nb_classes_inference"] = 11 
        config["target_dataset_name"] = 'SemanticKITTI'
        config["target_dataset_root"] = 'data/SemanticKITTI'
    elif config["parameter"]["setting"] == "NS2NS":
        config["source_dataset_name"] = 'NuScenes'
        config["source_dataset_root"] = 'data/nuscenes'
        config["nb_classes"] = 17
        config["nb_classes_inference"] = 17
        config["target_dataset_name"] = 'NuScenes'
        config["target_dataset_root"] = 'data/nuscenes'

    elif config["parameter"]["setting"] == "NS2POSS":
        config["source_dataset_name"] = 'NuScenes'
        config["source_dataset_root"] = 'data/nuscenes'
        config["nb_classes"] = 17
        config["nb_classes_inference"] = 7 
        config["target_dataset_name"] = 'SemanticPOSS'
        config["target_dataset_root"] = 'data/SemanticPOSS'

    elif config["parameter"]["setting"] == "Synth2SK":
        config["source_dataset_name"] = 'SynLidar'
        config["source_dataset_root"] = 'data/synlidar'
        config["nb_classes"] = 23
        config["nb_classes_inference"] = 20 
        config["target_dataset_name"] = 'SemanticKITTI'
        config["target_dataset_root"] = 'data/SemanticKITTI'
    elif config["parameter"]["setting"] == "Synth2POSS":
        config["source_dataset_name"] = 'SynLidar'
        config["source_dataset_root"] = 'data/synlidar'
        config["nb_classes"] = 23
        config["nb_classes_inference"] = 14 
        config["target_dataset_name"] = 'SemanticPOSS'
        config["target_dataset_root"] = 'data/SemanticPOSS'
    elif config["parameter"]["setting"] == "NS2PD":
        config["source_dataset_name"] = 'NuScenes'
        config["source_dataset_root"] = 'data/nuscenes'
        config["nb_classes"] = 17
        config["nb_classes_inference"] = 9 
        config["target_dataset_name"] = 'Pandaset'
        config["target_dataset_root"] = 'data/pandaset'
    elif config["parameter"]["setting"] == "NS2WY":
        config["source_dataset_name"] = 'NuScenes'
        config["source_dataset_root"] = 'data/nuscenes'
        config["nb_classes"] = 17
        config["nb_classes_inference"] = 11 
        config["target_dataset_name"] = 'Waymo'
        config["target_dataset_root"] = 'data/waymo'
    else:
        raise ValueError(f"Setting: {config['parameter']['setting']} not available")
    
    return config


def inference_merger(training_map, inference_mapping, ignore_index=0): 
    training_values = np.array(list(training_map.values()))
    inference_values = np.array(list(inference_mapping.values()))
    
    training_label = np.unique(training_values)
    inference_label = np.unique(inference_values)
    mapping_to_ignore = None
    merging_dict = {}
    for label_nb in inference_label: 
        indices = label_nb==inference_values
        
        #Indexes of training predictions, that should be mapped to the inference label label_nb
        c = np.unique(training_values[indices])
        if label_nb == ignore_index:
            mapping_to_ignore = c
        
        for i in c:
            if i == ignore_index and not label_nb == ignore_index: 
                continue
            if i in merging_dict: 
                raise ("Ambiguity in mapping")
            else:
                merging_dict[i]=label_nb
    
    #Check that all indices of training are taken
    assert len(np.intersect1d(np.unique(list(merging_dict.keys())),training_values)) == len(training_label)
    
    return {"merging_dict":merging_dict, "mapping_to_ignore": mapping_to_ignore}

def sf_class_mapping_loader(source_dataset="NuScenes", target_dataset="SemanticKITTI"):
    mapping_information = None
    
    # Loading the mappings for NuScenes
    if source_dataset == "NuScenes":
        ns_pretrained_mapping = read_yaml_file("utils_source_free/nuscenes_mapping.yaml")
        ns_training_map = ns_pretrained_mapping["single"]
        ns_inference_mapping_sk_cl = ns_pretrained_mapping["semantic_kitti_cl"]
        ns_inference_mapping_poss = ns_pretrained_mapping["semantic_poss"]
        ns_inference_mapping_panda = ns_pretrained_mapping["pandaset"]
        ns_inference_mapping_waymo = ns_pretrained_mapping["waymo"]
        
        if target_dataset == "SemanticKITTI":
            print("Mapping NS2SK")
            mapping_information = inference_merger(ns_training_map, ns_inference_mapping_sk_cl)
        elif target_dataset == "SemanticPOSS":
            print("Mapping NS2POSS")
            mapping_information = inference_merger(ns_training_map, ns_inference_mapping_poss)
        elif  target_dataset == "NuScenes" or target_dataset == "NuScenes_C":
            print("Mapping NS2NS")
            mapping_information = inference_merger(ns_training_map, ns_training_map)
        elif target_dataset == "Pandaset":
            print("Mapping NS2PD")
            mapping_information = inference_merger(ns_training_map, ns_inference_mapping_panda)
        elif target_dataset == "Waymo":
            print("Mapping NS2WY")
            mapping_information = inference_merger(ns_training_map, ns_inference_mapping_waymo)
            
        else: 
            raise ValueError(f"Unknown combination of source: {source_dataset} and target {target_dataset}")
        
        
        print("Ambiguity Mapping Testing:")
        try: 
            _ = inference_merger(synth_training_map, synth_test)
        except: 
            print("Correctly raising an error for wrongly mapping")
        
    elif source_dataset == "SynLidar": 
        
        synth_pretrained_mapping = read_yaml_file("utils_source_free/synth_mapping.yaml")

        synth_training_map = synth_pretrained_mapping["single"]
        synth_inference_mapping_sk = synth_pretrained_mapping["semantic_kitti"]
        synth_inference_mapping_poss = synth_pretrained_mapping["semantic_poss"]
        synth_test = synth_pretrained_mapping["ambiguity_test"]
        if target_dataset == "SemanticKITTI":
            print("Mapping Synth2SK")
            mapping_information = inference_merger(synth_training_map, synth_inference_mapping_sk)
        elif target_dataset == "SemanticPOSS":
            print("Mapping Synth2POSS")
            mapping_information = inference_merger(synth_training_map, synth_inference_mapping_poss)
        elif  target_dataset == "Synlidar":
            print("Mapping Synth2Synth")
            mapping_information = inference_merger(synth_training_map, synth_training_map)
        else: 
            raise (f"Unknown combination of source: {source_dataset} and target {target_dataset}")
        
        
        print("Ambiguity Mapping Testing:")
        try: 
            _ = inference_merger(synth_training_map, synth_test)

        except: 
            print("Correctly raising an error for wrongly mapping")
    
    elif source_dataset == "SemanticKITTI": 
        
        sk_pretrained_mapping = read_yaml_file("utils_source_free/semantickitti_mapping.yaml")

        sk_training_map = sk_pretrained_mapping["single"]
        sk_test = sk_pretrained_mapping["ambiguity_test"]
        
        if target_dataset == "SemanticKITTI":
            print("Mapping SK2SK")
            mapping_information = inference_merger(sk_training_map, sk_training_map)
        
        else: 
            raise (f"Unknown combination of source: {source_dataset} and target {target_dataset}")
        
        
        print("Ambiguity Mapping Testing:")
        try: 
            _ = inference_merger(synth_training_map, synth_test)
        except: 
            print("Correctly raising an error for wrongly mapping")
    else: 
        raise ("Unknown source dataset")
    return mapping_information

def prediction_changer(prediction_matrix, merging_information, after_softmax=False, validation=True, numpy_return=True):
    """
    prediction_matrix (Float Tensor NxC): predictions
    merging_information (dict): information about merging
    after_softmax (bool=False): flag to indicate if softmax is already applied  
    
    N: number of predictions
    C: number of classes (network classes, unmapped)
    """
    
    #Evaluation prediction, set the ingored classes to -inf
    mapping_to_ignore = merging_information["mapping_to_ignore"]
    mapping_array = torch.zeros((np.max([k for k in  merging_information["merging_dict"].keys()]) + 1), dtype=torch.int)
            
    for k, v in merging_information["merging_dict"].items():
        mapping_array[k] = v
    min_value = torch.finfo(prediction_matrix.dtype).min
    prediction_matrix[:, mapping_to_ignore]=min_value
    pred = torch.argmax(prediction_matrix, dim=1)
    mapped_pred = mapping_array[pred]
    if numpy_return:
        return mapped_pred.numpy()
    else: 
        return mapped_pred
    
def prediction_mapper(pred, merging_information, numpy_return=True):
    """
    prediction (Float Tensor N): predictions
    merging_information (dict): information about merging
    N: number of predictions
    
    """
    mapping_array = torch.zeros((np.max([k for k in  merging_information["merging_dict"].keys()]) + 1), dtype=torch.int)
            
    for k, v in merging_information["merging_dict"].items():
        mapping_array[k] = v

    mapped_pred = mapping_array[pred]
    if numpy_return:
        return mapped_pred.numpy()
    else: 
        return mapped_pred
    

def validation_non_premap(net, config, test_loader, epoch, disable_log, device, list_ignore_classes=[]):
    net.eval()
    
    
    cm_seg_head = np.zeros((config["nb_classes_inference"],config["nb_classes_inference"]))
    mapping_information = sf_class_mapping_loader(source_dataset=config["source_dataset_name"], target_dataset=config["target_dataset_name"])
    with torch.no_grad():
        count_iter = 0
        t = tqdm(test_loader,desc="  Test " + str(epoch),ncols=200,disable=disable_log,)
        for data in t:

            data = dict_to_device(data, device)
            
            _, output_seg, _ = net.forward_mapped_learned(data) #SOURCE data
            
            
            output_seg_np = prediction_changer(output_seg.cpu().detach(), mapping_information)#np.argmax(output_seg[:,1:].cpu().detach().numpy(), axis=1) + 1 ###As 0 is ignored, we always look for the highest values in the non 0 part --> have to add + 1 at the index
            target_seg_np = data["y"].cpu().numpy().astype(int)
            cm_seg_head_ = confusion_matrix(target_seg_np.ravel(), output_seg_np.ravel(), labels=list(range(config["nb_classes_inference"])))
            cm_seg_head += cm_seg_head_
            count_iter += 1
            if count_iter % 10 == 0:
                torch.cuda.empty_cache()
        cm_seg_head_with_ignore = np.copy(cm_seg_head)
        test_seg_head_maa, accuracy_per_class = metrics.stats_accuracy_per_class(cm_seg_head, ignore_list=list_ignore_classes) #First return value is the mean IoU
        test_seg_head_miou, seg_iou_per_class = metrics.stats_iou_per_class(cm_seg_head, ignore_list=list_ignore_classes) #First return value is the mean IoU
            

    return_data = {"test_seg_head_miou":test_seg_head_miou,\
            "test_seg_head_maa":test_seg_head_maa, "test_seg_head_loss":None, "seg_iou_per_class":seg_iou_per_class,\
                "accuracy_per_class":accuracy_per_class, "cm_seg_head":cm_seg_head, "cm_seg_head_with_ignore":cm_seg_head_with_ignore}
    return return_data




def summation_matrix_generator(mapping_information):
        merging_dict = mapping_information["merging_dict"]
        max_value_keys = max([key for key in merging_dict.keys() if isinstance(key, np.int64)])
        min_value_keys = min([key for key in merging_dict.keys() if isinstance(key, np.int64)])

        max_value_items = max([key for key in merging_dict.values() if isinstance(key, np.int64)])
        min_value_items = min([key for key in merging_dict.values() if isinstance(key, np.int64)])
        
        number_keys = len(list(merging_dict.keys()))
        number_values = len(np.unique(list(merging_dict.values())))
        
        assert max_value_keys + 1 == number_keys
        assert max_value_items + 1 == number_values
        assert min_value_keys == 0 
        assert min_value_items == 0 
        
        
        #Array that you multiplicate with for the merging
        merger_array = torch.zeros((number_keys, number_values))
        
        #Indexes to set to 1 
        list_position_index = merging_dict.items()
        for tupple_position_index in list_position_index:
            row, column = tupple_position_index
            merger_array[row, column] = 1 
        return merger_array

def fast_hist(pred, label, n):
    assert torch.all(label > -1) & torch.all(pred > -1)
    assert torch.all(label < n) & torch.all(pred < n)
    return torch.bincount(n * label + pred, minlength=n**2).reshape(n, n)


def per_class_iu(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def overall_accuracy(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist).sum() / hist.sum()


def per_class_accuracy(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist) / hist.sum(1)