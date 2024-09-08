import glob
import importlib
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch._C import Value
from torch_geometric.data import Data, Dataset


import logging
from os import listdir
from os.path import exists, isdir, join
from pathlib import Path

import yaml
from scipy.spatial import cKDTree


def class_mapping_da(config):
    if config["source_dataset_name"] == "NuScenes" and  config["target_dataset_name"] == "Waymo":
        return {
            0:0,#undefined->background/noise
            1:1,#car->Car
            2:4,#truck->Truck
            3:5,#bus->Other-Vehicle
            4:5,#other-vehicle->Other-Vehicle
            5:0,#motorcyclist->background/noise
            6:0,#bicyclist->background/noise
            7:6,#pedestrian->Pedestrian
            8:0,#sign->background/noise
            9:0,#traffic-light->background/noise
            10:0,#pole->background/noise
            11:0,#construction-cone->background/noise
            12:2,#bicycle->Bicycle
            13:3,#motorcycle->Motorcycle
            14:0,#building->background/noise
            15:10,#vegetation, ->Vegetation
            16:10,#trunk->Vegetation
            17:8,#curb->Sidewalk
            18:7,#road->Driveable-surface
            19:7,#lane-marker->Driveable-surface
            20:7,#other-ground->Driveable-surface
            21:9,#walkable->Walkable
            22:8,#sidewalk->Sidewalk
        }
    elif config["source_dataset_name"] == "Waymo" and  config["target_dataset_name"] == "Waymo":
        return {
            0:0,#undefined
            1:1,#car
            2:2,#truck
            3:3,#bus
            4:4,#other-vehicle
            5:5,#motorcyclist
            6:6,#bicyclist
            7:7,#pedestrian
            8:8,#sign
            9:9,#traffic-light
            10:10,#pole
            11:11,#construction-cone
            12:12,#bicycle
            13:13,#motorcycle
            14:14,#building
            15:15,#vegetation
            16:16,#trunk
            17:17,#curb
            18:18,#road
            19:19,#lane-marker
            20:20,#other-ground
            21:21,#walkable
            22:22,#sidewalk
        }
    else: 
        raise ValueError(f"No mapping for {config['source_dataset_name']} to { config['target_dataset_name']}")

class Waymo(Dataset):

    CLASS_NAME = [
        "car",
        "truck",
        "bus",
        "other_vehicle",
        "motorcyclist",
        "bicyclist",
        "pedestrian",
        "sign",
        "traffic_light",
        "pole",
        "construction_cone",
        "bicycle",
        "motorcycle",
        "building",
        "vegetation",
        "tree_trunk",
        "curb",
        "road",
        "lane_marker",
        "other_ground",
        "walkable",
        "sidewalk",
    ]

    def __init__(self,
                 root,
                 split="training",
                 transform=None, 
                 dataset_size=None,
                 multiframe_range=None,
                 skip_ratio=1,
                 da_flag =False, config=None,
                 **kwargs):
        super().__init__(root, transform, None)

        self.split = split
        self.root = root
        self.da_flag = da_flag
        self.config = config
        self.N_LABELS = self.config["nb_classes"] if self.config is not None else 11
        self.files = []
        if self.split == "train":
            for file in glob.glob(os.path.join(self.root, "training","*", "lidar", "*_r1.npy"), recursive=True):
                self.files.append(file)
        elif self.split == "val":
            for file in glob.glob(os.path.join(self.root, "validation","*", "lidar", "*_r1.npy"), recursive=True):
                self.files.append(file)
        else:
            raise ValueError

        print(f"Waymo dataset - {self.split} - {len(self.files)}")

        learning_map = class_mapping_da(config)
        self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
        for k, v in learning_map.items():
            self.learning_map[k] = v
        
    def __len__(self):
        return len(self.files)

    def get(self, index):
        # Load point cloud
        pc1 = np.load(self.files[index])
        pc2 = np.load(str(self.files[index]).replace("_r1.npy", "_r2.npy"))
        pc = np.concatenate([pc1, pc2], axis=0)
        #Get the positions 
        pos = pc[:,:3]
        # process the point cloud
        intensities = pc[:,3]
        intensities[intensities > 1] = 1
        intensities[intensities < 0] = 0
        #pc[:,3] = intensities

        # Extract Label
        label_filename = str(self.files[index]).replace("/lidar/", "/labels/")
        labels1 = np.load(label_filename).astype(np.int32)
        labels2 = np.load(label_filename.replace("_r1.npy", "_r2.npy")).astype(np.int32)
        y = np.concatenate([labels1, labels2], axis=0)
        y = self.learning_map[y]
        #labels = labels - 1
        #labels[labels==-1] = 255
        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        intensities = torch.tensor(intensities, dtype=torch.float)
        x = torch.ones((pos.shape[0],1), dtype=torch.float)
        
        #Number of points in this scene
        N = torch.from_numpy(np.array(pos.shape[0]))[None]
        #Scene index
        shape_id = torch.from_numpy(np.array(index))[None]
        return Data(x=x, intensities=intensities, pos=pos, y=y, shape_id=shape_id, N=N)
    
    
    def get_weights(self):
            weights = torch.ones(self.N_LABELS)
            weights[0] = 0
            return weights

    @staticmethod
    def get_mask_filter_valid_labels(y):
        return (y>0)

    @property
    def raw_file_names(self):
        return []

    def _download(self): # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.files)

    def get_category(self, f_id):
        return str(self.files[f_id]).split("/")[-3]

    def get_object_name(self, f_id):
        return str(self.files[f_id]).split("/")[-1]

    def get_class_name(self, f_id):
        return "lidar"

    def get_save_dir(self, f_id):
        return os.path.join(str(self.files[f_id]).split("/")[-3], str(self.files[f_id]).split("/")[-2])

    def get_filename(self, idx):
        return self.files[idx]