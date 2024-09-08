
import glob
import importlib
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset

import logging
import os.path as osp
import sys
from functools import reduce
from pathlib import Path

# Basic libs
import numpy as np
import yaml
from nuscenes import NuScenes as NuScenes_
from nuscenes.utils.data_classes import LidarPointCloud, LidarSegPointCloud
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from scipy.spatial import cKDTree


def class_mapping_ns():
    return {0: 'noise',
            1: 'animal',
            2: 'human.pedestrian.adult',
            3: 'human.pedestrian.child',
            4: 'human.pedestrian.construction_worker',
            5: 'human.pedestrian.personal_mobility',
            6: 'human.pedestrian.police_officer',
            7: 'human.pedestrian.stroller',
            8: 'human.pedestrian.wheelchair',
            9: 'movable_object.barrier',
            10: 'movable_object.debris',
            11: 'movable_object.pushable_pullable',
            12: 'movable_object.trafficcone',
            13: 'static_object.bicycle_rack',
            14: 'vehicle.bicycle',
            15: 'vehicle.bus.bendy',
            16: 'vehicle.bus.rigid',
            17: 'vehicle.car',
            18: 'vehicle.construction',
            19: 'vehicle.emergency.ambulance',
            20: 'vehicle.emergency.police',
            21: 'vehicle.motorcycle',
            22: 'vehicle.trailer',
            23: 'vehicle.truck',
            24: 'flat.driveable_surface',
            25: 'flat.other',
            26: 'flat.sidewalk',
            27: 'flat.terrain',
            28: 'static.manmade',
            29: 'static.other',
            30: 'static.vegetation',
            31: 'vehicle.ego'
            }
def class_mapping_da(config):
    # Adaption of mapping of classes in DA Case
    

    if config["source_dataset_name"] == "NuScenes" and  config["target_dataset_name"] == "NuScenes": 
        # Original NS semantic Segmentation mapping
        if config["nb_classes"] == 7:
            return {
                0:0,#noise-->ignore 
                1:0,#animal-->ignore 
                2:0,#human.pedestrian.adult-->ignore 
                3:0,#human.pedestrian.child-->ignore 
                4:0,#human.pedestrian.construction_worker-->ignore 
                5:0,#human.pedestrian.personal_mobility-->ignore 
                6:0,#human.pedestrian.police_officer-->ignore 
                7:0,#human.pedestrian.stroller-->ignore 
                8:0,#human.pedestrian.wheelchair-->ignore 
                9:0,#movable_object.barrier-->ignore 
                10:0,#movable_object.debris-->ignore 
                11:0,#movable_object.pushable_pullable-->ignore 
                12:0,#movable_object.trafficcone-->ignore 
                13:0,#static_object.bicycle_rack-->ignore 
                14:1,#vehicle.bicycle-->vehicle
                15:1,#vehicle.bus.bendy-->vehicle
                16:1,#vehicle.bus.rigid-->vehicle
                17:1,#vehicle.car-->vehicle
                18:1,#vehicle.construction-->vehicle
                19:0,#vehicle.emergency.ambulance-->ignore 
                20:0,#vehicle.emergency.police-->ignore 
                21:1,#vehicle.motorcycle-->vehicle
                22:1,#vehicle.trailer-->vehicle
                23:1,#vehicle.truck-->vehicle
                24:2,#flat.driveable_surface-->driveable_surface
                25:0,#flat.other-->ignore 
                26:3,#flat.sidewalk-->sidewalk
                27:4,#flat.terrain-->terrain
                28:5,#static.manmade-->manmade
                29:0,#static.other-->ignore 
                30:6,#static.vegetation-->vegetation
                31:0,#vehicle.ego-->ignore 
            }
        
        else: 
            return {
                0:0,
                1:0,
                2:7,
                3:7,
                4:7,
                5:0,
                6:7,
                7:0,
                8:0,
                9:1,
                10:0,
                11:0,
                12:8,
                13:0,
                14:2,
                15:3,
                16:3,
                17:4,
                18:5,
                19:0,
                20:0,
                21:6,
                22:9,
                23:10,
                24:11,
                25:12,
                26:13,
                27:14,
                28:15,
                29:0,
                30:16,
                31:0
            }
    elif config["source_dataset_name"] == "SemanticKITTI" or  config["target_dataset_name"] == "SemanticKITTI":
        return {
        0: 0,
        1: 0,
        2: 6,
        3: 6,
        4: 6,
        ###Changes 
        5: 6, #Prev 0
        6: 6,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 0,
        13: 0,
        14: 2,
        15: 5,
        16: 5,
        17: 1,
        18: 5,
        19: 5, #Prev 0
        20: 5, #Prev 0
        21: 3,
        22: 5,
        23: 4,
        24: 7,
        25: 0,
        26: 8,
        27: 9,
        28: 0,
        29: 0,
        30: 10,
        31: 0,
        }
    

    elif config["source_dataset_name"] == "Livox" or  config["target_dataset_name"] == "Livox":
        return {
        0: 0,
        1: 7, #dog
        2: 6, #Pedestrian
        3: 6, #Pedestrian
        4: 6, #Pedestrian
        5: 6, #Pedestrian
        6: 6, #Pedestrian
        7: 6, #Pedestrian
        8: 6, #Pedestrian
        9: 0,
        10: 0,
        11: 0,
        12: 0,
        13: 0,
        14: 4, #Bicycle
        15: 3,#Bus
        16: 3, #Bus
        17: 1, #Car
        18: 0,
        19: 0, 
        20: 0, 
        21: 5, #Motorcycle
        22: 0,
        23: 2, #Truck
        24: 8, #Road
        25: 0,
        26: 0,
        27: 0,
        28: 9, #Building
        29: 0,
        30: 0,
        31: 0,
        }
    elif config["source_dataset_name"] == "Pandaset" or  config["target_dataset_name"] == "Pandaset":
        return {0:	0,#	noise --> Noise
        1:	0,#	animal --> Noise
        2:	2,#	human.pedestrian.adult --> pedestrian
        3:	2,#	human.pedestrian.child --> pedestrian
        4:	2,#	human.pedestrian.construction_worker --> pedestrian
        5:	0,#	human.pedestrian.personal_mobility --> Noise
        6:	2,#	human.pedestrian.police_officer --> pedestrian
        7:	0,#	human.pedestrian.stroller --> Noise
        8:	0,#	human.pedestrian.wheelchair --> Noise
        9:	6,#	movable_object.barrier --> manmade
        10:	0,#	movable_object.debris --> Noise
        11:	0,#	movable_object.pushable_pullable --> Noise
        12:	6,#	movable_object.trafficcone --> manmade
        13:	0,#	static_object.bicycle_rack --> Noise
        14:	1,#	vehicle.bicycle --> 2-wheeled
        15:	8,#	vehicle.bus.bendy --> 4-wheeled
        16:	8,#	vehicle.bus.rigid --> 4-wheeled
        17:	8,#	vehicle.car --> 4-wheeled
        18:	8,#	vehicle.construction --> 4-wheeled
        19:	0,#	vehicle.emergency.ambulance --> Noise
        20:	0,#	vehicle.emergency.police --> Noise
        21:	1,#	vehicle.motorcycle --> 2-wheeled
        22:	8,#	vehicle.trailer --> 4-wheeled
        23:	8,#	vehicle.truck --> 4-wheeled
        24:	3,#	flat.driveable_surface --> driveable ground
        25:	5,#	flat.other --> other ground
        26:	4,#	flat.sidewalk --> sidewalk
        27:	5,#	flat.terrain --> other ground
        28:	6,#	static.manmade --> manmade
        29:	0,#	static.other --> Noise
        30:	7,#	static.vegetation --> vegetation
        31:	0}#	vehicle.ego --> Noise
    
    elif config["target_dataset_name"] == "SemanticPOSS":
        return {0:0,#noise-->unlabeled
                1:0,#animal-->unlabeled
                2:1,#human.pedestrian.adult-->person
                3:1,#human.pedestrian.child-->person
                4:1,#human.pedestrian.construction_worker-->person
                5:0,#human.pedestrian.personal_mobility-->unlabeled
                6:1,#human.pedestrian.police_officer-->person
                7:0,#human.pedestrian.stroller-->unlabeled
                8:0,#human.pedestrian.wheelchair-->unlabeled
                9:6,#movable_object.barrier-->manmade
                10:0,#movable_object.debris-->unlabeled
                11:0,#movable_object.pushable_pullable-->unlabeled
                12:6,#movable_object.trafficcone-->manmade
                13:0,#static_object.bicycle_rack-->unlabeled
                14:2,#vehicle.bicycle-->bike
                15:3,#vehicle.bus.bendy-->car
                16:3,#vehicle.bus.rigid-->car
                17:3,#vehicle.car-->car
                18:3,#vehicle.construction-->car
                19:0,#vehicle.emergency.ambulance-->unlabeled
                20:0,#vehicle.emergency.police-->unlabeled
                21:2,#vehicle.motorcycle-->bike
                22:3,#vehicle.trailer-->car
                23:3,#vehicle.truck-->car
                24:4,#flat.driveable_surface-->ground
                25:4,#flat.other-->ground
                26:4,#flat.sidewalk-->ground
                27:4,#flat.terrain-->ground
                28:6,#static.manmade-->manmade
                29:0,#static.other-->unlabeled
                30:5,#static.vegetation-->vegetation
                31:0 }#vehicle.ego-->unlabeled
    elif config["source_dataset_name"] == "Waymo" or  config["target_dataset_name"] == "Waymo":
        return {
            0:0,#noise, animal, human.pedestrian.personal.mobility, ->background/noise
            1:0,#noise, animal, human.pedestrian.personal.mobility, ->background/noise
            2:6,#human.pedestrian.adult,->Pedestrian
            3:6,#human.pedestrian.adult,->Pedestrian
            4:6,#human.pedestrian.adult,->Pedestrian
            5:0,#noise, animal, human.pedestrian.personal.mobility, ->background/noise
            6:6,#human.pedestrian.adult,->Pedestrian
            7:0,#noise, animal, human.pedestrian.personal.mobility, ->background/noise
            8:0,#noise, animal, human.pedestrian.personal.mobility, ->background/noise
            9:0,#barrier->background/noise
            10:0,#noise, animal, human.pedestrian.personal.mobility, ->background/noise
            11:0,#noise, animal, human.pedestrian.personal.mobility, ->background/noise
            12:0,#traffic_cone->background/noise
            13:0,#noise, animal, human.pedestrian.personal.mobility, ->background/noise
            14:2,#vehicle.bicycle->Bicycle
            15:5,#vehicle.bus.bendy->Other-Vehicle
            16:5,#vehicle.bus.bendy->Other-Vehicle
            17:1,#vehicle.car->Car
            18:5,#vehicle.construction,->Other-Vehicle
            19:0,#noise, animal, human.pedestrian.personal.mobility, ->background/noise
            20:0,#noise, animal, human.pedestrian.personal.mobility, ->background/noise
            21:3,#vehicle.motorcycle->Motorcycle
            22:5,#vehicle.trailer->Other-Vehicle
            23:4,#vehicle.truck->Truck
            24:7,#flat.driveable surface->Driveable-surface
            25:0,#other_flat->background/noise
            26:8,#flat.sidewalk->Sidewalk
            27:9,#flat.terrain->Walkable
            28:0,#manmade->background/noise
            29:0,#noise, animal, human.pedestrian.personal.mobility, ->background/noise
            30:10,#static.vegetation->Vegetation
            31:0,#noise, animal, human.pedestrian.personal.mobility, ->background/noise
            }
    else: 
        raise ValueError("No mapping")

class NuScenes(Dataset):

    N_LABELS=11

    def __init__(self,
                 root,
                 split="training",
                 transform=None,
                 skip_ratio=1,
                 da_flag = False, config=None,
                 **kwargs):

        super().__init__(root, transform, None)

        self.config = config
        self.nusc = NuScenes_(version=self.config['ns_dataset_version'], dataroot=self.root, verbose=True)
        # self.nusc = NuScenes_(version='v1.0-mini', dataroot=self.root, verbose=True)

        self.da_flag = da_flag
        logging.info("Nuscenes dataset - creating splits")

        # get the scenes
        assert(split in ["train", "val", "test", "verifying", "parametrizing", "ssda"])
        if split == "ssda": 
            split="train"
        
        if split == "verifying":
            phase_scenes = CUSTOM_SPLIT
        elif split == "parametrizing":
            phase_scenes = list( set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT) )
        else:
            phase_scenes = create_splits_scenes()[split]


        # create a list of camera & lidar scans
        skip_counter = 0
        self.list_keyframes = []
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                location = self.nusc.get("log", scene['log_token'])['location']
                #Filtering for locations
                if ("no_location_filter" in self.config and (self.config["no_location_filter"] or self.config["{}_location".format(kwargs["domain"])] in location)) \
                    or (not "no_location_filter" in self.config):
                    
                    if skip_counter % skip_ratio == 0:
                        current_sample_token = scene["first_sample_token"]

                        # Loop to get all successive keyframes
                        list_data = []
                        while current_sample_token != "":
                            current_sample = self.nusc.get("sample", current_sample_token)
                            list_data.append(current_sample["data"])
                            current_sample_token = current_sample["next"]

                        # Add new scans in the list
                        self.list_keyframes.extend(list_data)

        

        print(f"Nuscnes dataset split {split} - {len(self.list_keyframes)} frames")

        self.label_to_name = class_mapping_ns()
        
        if self.da_flag: 
            self.label_to_reduced = class_mapping_da(self.config)
        
        self.label_to_reduced_np = np.zeros(32, dtype=np.int)
        for i in range(32):
            self.label_to_reduced_np[i] = self.label_to_reduced[i]
        

    def get_weights(self):
        weights = torch.ones(self.N_LABELS)
        weights[0] = 0
        return weights

    @staticmethod
    def get_mask_filter_valid_labels(y):
        return (y>0)

    @staticmethod
    def get_ignore_index():
        return 0

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
        return len(self.list_keyframes)

    def get_category(self, f_id):
        raise NotImplementedError

    def get_object_name(self, f_id):
        raise NotImplementedError

    def get_class_name(self, f_id):
        raise NotImplementedError

    def get_save_dir(self, f_id):
        raise NotImplementedError


    def get(self, idx):
        """Get item."""

        data = self.list_keyframes[idx]

        # get the lidar
        lidar_token = data['LIDAR_TOP']
        lidar_rec = self.nusc.get('sample_data', lidar_token)
        pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_rec['filename']))
        pc = pc.points.T

        pos = pc[:,:3]
        if "no_scaling_intensities" in self.config and self.config["no_scaling_intensities"]: 
            #Probably very harmful and not recommended
            intensities = pc[:,3:]  # intensities not scaled
            #print("Intensities scaled")
        else:
            intensities = pc[:,3:] / 255 # intensities

        # get the labels
        lidarseg_label_filename = os.path.join(self.nusc.dataroot, self.nusc.get('lidarseg', lidar_token)['filename'])
        y_complete_labels = load_bin_file(lidarseg_label_filename)

        y = self.label_to_reduced_np[y_complete_labels]
        
        #mask = np.linalg.norm(pos, axis=1)<50
        #pos = pos[mask]
        #y = y[mask]
        #intensities = intensities[mask]

        # convert to torch
        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        intensities = torch.tensor(intensities, dtype=torch.float)
        x = torch.ones((pos.shape[0],1), dtype=torch.float)
        #Number of points in this scene
        N = torch.from_numpy(np.array(pos.shape[0]))[None]
        #Scene index
        shape_id = torch.from_numpy(np.array(idx))[None]
        
        if "input_sem" in self.config and self.config["input_sem"]: 
            one_hot_semant = F.one_hot(y, num_classes=11)
            stacked = torch.hstack((x,one_hot_semant))
            x = stacked
        return Data(x=x, intensities=intensities, pos=pos, y=y, shape_id=shape_id, N=N)