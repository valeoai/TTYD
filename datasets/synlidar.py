import glob
import importlib
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch._C import Value
from torch_geometric.data import Data, Dataset


import logging
import sys
from pathlib import Path

# Basic libs
import numpy as np
import yaml
from scipy.spatial import cKDTree


class SynLidar(Dataset):

    def __init__(self,
                 root,
                 split="training",
                 transform=None, 
                 dataset_size=None,
                 skip_ratio=1,
                 da_flag =False, config=None,
                 **kwargs):
        
        super().__init__(root, transform, None)

        self.split = split
        self.n_frames = 1
        self.da_flag = da_flag
        self.config = config

        
        logging.info(f"SynLidar - split {split}")

        # get the scenes
        assert(split in ["train", "val", "test", "verifying", "parametrizing"])
        if split == "verifying":
            self.sequences = ['{:02d}'.format(i) for i in range(13) if i == 7]
        elif split == "parametrizing":
            self.sequences = ['{:02d}'.format(i) for i in range(13) if (i != 8 and i != 7)]
        elif split == "train":
            self.sequences = ['{:02d}'.format(i) for i in range(13)]
        elif split == "val":
            self.sequences = ['{:02d}'.format(i) for i in range(13) if i == 8] ##Validation set is seen during training !!!
        elif split == "test":
            raise ValueError('Unknown set for SynLidar data: ', split)
        else:
            raise ValueError('Unknown set for SynLidar data: ', split)


        # get the filenames
        self.all_files = []
        for sequence in self.sequences:
            self.all_files += [path for path in Path(os.path.join(self.root, "SubDataset", "sequences", sequence, "velodyne")).rglob('*.bin')]
        
        ##Sort for verifying and parametrizing 
        if split == "verifying" or split == "val" or split == "parametrizing": 
            self.all_files = sorted(self.all_files, key=lambda i:str(i).lower())
        
        self.all_files = self.all_files[::skip_ratio]

        self.all_labels = []
        for fname in self.all_files:
            fname = str(fname).replace("/velodyne/", "/labels/")
            fname = str(fname).replace(".bin", ".label")
            self.all_labels.append(fname)


        # Read labels
        if self.n_frames == 1:
            config_file = os.path.join(self.root, 'semantic-kitti.yaml')
        elif self.n_frames > 1:
            config_file = os.path.join(self.root, 'semantic-kitti-all.yaml')
        else:
            raise ValueError('number of frames has to be >= 1')

        
        labels_names={0 : "unlabeled",
            1: "car",
            2: "pick-up",
            3: "truck",
            4: "bus",
            5: "bicycle",
            6: "motorcycle",
            7: "other-vehicle",
            8: "road",
            9: "sidewalk",
            10: "parking",
            11: "other-ground",
            12: "female",
            13: "male",
            14: "kid",
            15: "crowd",  # multiple person that are very close
            16: "bicyclist",
            17: "motorcyclist",
            18: "building",
            19: "other-structure",
            20: "vegetation",
            21: "trunk",
            22: "terrain",
            23: "traffic-sign",
            24: "pole",
            25: "traffic-cone",
            26: "fence",
            27: "garbage-can",
            28: "electric-box",
            29: "table",
            30: "chair",
            31: "bench",
            32: "other-object"}


        
            
        if self.da_flag: 
            #Changes of mapping in DA case
            if self.config["source_dataset_name"] == "SemanticKITTI" or  self.config["target_dataset_name"] == "SemanticKITTI":
                learning_map={
                    0: 0,  # "unlabeled"
                    1: 1,  # "car"
                    2: 4,  # "pick-up"
                    3: 4,  # "truck"
                    4: 5,  # "bus"
                    5: 2,  # "bicycle"
                    6: 3,  # "motorcycle"
                    7: 5,  # "other-vehicle"
                    8: 9,  # "road"
                    9: 11,  # "sidewalk"
                    10: 10,  # "parking"
                    11: 12,  # "other-ground"
                    12: 6, # "female"
                    13: 6,  # "male"
                    14: 6,  # "kid"
                    15: 6,  # "crowd"
                    16: 7,  # "bicyclist"
                    17: 8,  # "motorcyclist"
                    18: 13,  # "building"
                    19: 0,  # "other-structure"
                    20: 15,  # "vegetation"
                    21: 16,  # "trunk"
                    22: 17,  # "terrain"
                    23: 19,  # "traffic-sign"
                    24: 18,  # "pole"
                    25: 0,  # "traffic-cone"
                    26: 14,  # "fence"
                    27: 0,  # "garbage-can"
                    28: 0,  # "electric-box"
                    29: 0,  # "table"
                    30: 0,  # "chair"
                    31: 0,  # "bench"
                    32: 0  # "other-object"
                }
                learning_map_inv ={ # inverse of previous map
                    0: 0,     
                    1: 1,     
                    2: 5,    
                    3: 6,     
                    4: 2,     
                    5: 7,     
                    6: 12,
                    7: 16,  
                    8: 17,
                    9: 8, 
                    10: 10, 
                    11: 9,
                    12:11,
                    13:18,
                    14:26,
                    15:20,
                    16:21,
                    17:22,
                    18:24,
                    19:23,
                
                    }

            elif self.config["source_dataset_name"] == "SemanticPOSS" or  self.config["target_dataset_name"] == "SemanticPOSS":
                learning_map={0: 0,  # "unlabeled"
                       1: 3,  # "car"
                    2: 0,  # "pick-up"
                    3: 0,  # "truck"
                    4: 0,  # "bus"
                    5: 12,  # "bicycle"
                    6: 0,  # "motorcycle"
                    7: 0,  # "other-vehicle"
                    8: 13,  # "road"
                    9: 0,  # "sidewalk"
                    10: 0,  # "parking"
                    11: 0,  # "other-ground"
                    12: 1,  # "female"
                    13: 1,  # "male"
                    14: 1,  # "kid"
                    15: 1,  # "crowd"
                    16: 2,  # "bicyclist"
                    17: 2,  # "motorcyclist"
                    18: 9,  # "building"
                    19: 0,  # "other-structure"
                    20: 5,  # "vegetation"
                    21: 4,  # "trunk"
                    22: 0,  # "terrain"
                    23: 6,  # "traffic-sign"
                    24: 7,  # "pole"
                    25: 10,  # "traffic-cone"
                    26: 11,  # "fence"
                    27: 8,  # "garbage-can"
                    28: 0,  # "electric-box"
                    29: 0,  # "table"
                    30: 0,  # "chair"
                    31: 0,  # "bench"
                    32: 0}  # "other-object"
            elif self.config["source_dataset_name"] == "SynLidar" and self.config["target_dataset_name"] == "SynLidar":
                learning_map={0:	0,#	unlabeled,	-->	Unlabeled
                    1:	1,#	car,	-->	Car 
                    2:	2,#	pick-up,	-->	Truck
                    3:	2,#	truck,	-->	Truck
                    4:	3,#	bus,	-->	Bus
                    5:	4,#	bicycle,	-->	Bicycle
                    6:	5,#	motorcycle,	-->	Motorcycle
                    7:	6,#	other-vehicle,	-->	Other Vehicle
                    8:	7,#	road,	-->	Road
                    9:	8,#	sidewalk,	-->	Sidewalk
                    10:	9,#	parking,	-->	Parking
                    11:	10,#	other-ground,	-->	Other ground 
                    12:	11,#	female,	-->	Person
                    13:	11,	#	male,	-->	Person
                    14:	11,	#	kid,	-->	Person
                    15:	11,	#	crowd,  # multiple person that are very close	-->	Person
                    16:	12,	#	bicyclist,	-->	Bicyclist
                    17:	13,	#	motorcyclist,	-->	Motorcyclist
                    18:	14,	#	building,	-->	Building
                    19:	0,	#	other-structure,	-->	Unlabeled
                    20:	15,	#	vegetation,	-->	Vegetation
                    21:	16,	#	trunk,	-->	Trunk
                    22:	17,	#	terrain,	-->	Terrain
                    23:	18,	#	traffic-sign,	-->	Traffic-Sign
                    24:	19,	#	pole,	-->	Pole
                    25:	20,	#	traffic-cone,	-->	Traffic-Cone
                    26:	21,	#	fence,	-->	Fence
                    27:	22,	#	garbage-can,	-->	Garbage Can
                    28:	0,	#	electric-box,	-->	Unlabeled
                    29:	0,	#	table,	-->	Unlabeled
                    30:	0,	#	chair,	-->	Unlabeled
                    31:	0,	#	bench,	-->	Unlabeled
                    32:	0}	#	other-object	-->	Unlabeled
            else: 
                raise ValueError("No mapping")
                





            # An example of class mapping from synlidar to semanticposs,
            # classes that are indistinguishable from single scan or inconsistent in
            # ground truth are mapped to their closest equivalent.
            map_2_semanticposs={0: 255, # "unlabeled"
            1: 2,   # "car"
            2: 2,  
            3: 2,
            4: 2,
            5: 11,  # "bike"
            6: 11,
            7: 255,
            8: 12, # "ground"
            9: 12,
            10: 12,
            11: 12,
            12: 0,  # "person"
            13: 0,
            14: 0,
            15: 0,
            16: 1, # "rider"
            17: 1,
            18: 8,  # "building"
            19: 255,
            20: 4,  # "plant"
            21: 3,  # "trunk"
            22: 4,
            23: 5,  # "traffic-sign"
            24: 6,  # "pole"
            25: 9,  # "cone/stone"
            26: 10, # "fence"
            27: 7,  # "trashcan"
            28: 255,  
            29: 255, 
            30: 255, 
            31: 255, 
            32: 255}

            sequences=["00","01","02","03","04","05","06","07","08","09","10","11","12"]
        else:
            with open(config_file, 'r') as stream:
                doc = yaml.safe_load(stream)
                all_labels = doc['labels']
            learning_map = doc['learning_map']
            #learning_map_inv = doc['learning_map_inv']
            
        self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
        
        for k, v in learning_map.items():
            self.learning_map[k] = v
        
        #self.learning_map_inv = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), dtype=np.int32)
        #for k, v in learning_map_inv.items():
        #    self.learning_map_inv[k] = v
          
        

     

        self.class_colors = np.array([[0, 0, 0],[0, 0, 255],[245, 150, 100],
                                    [245, 230, 100],[250, 80, 100],[150, 60, 30],[255, 0, 0],
                                    [180, 30, 80],[255, 0, 0],[30, 30, 255],[200, 40, 255],
                                    [90, 30, 150],[255, 0, 255],[255, 150, 255],
                                    [75, 0, 75],[75, 0, 175],[0, 200, 255],
                                    [50, 120, 255],[0, 150, 255],[170, 255, 150],
                                    [0, 175, 0],[0, 60, 135],[80, 240, 150],
                                    [150, 240, 255],[0, 0, 255],[255, 255, 50],
                                    [245, 150, 100],[255, 0, 0],[200, 40, 255],
                                    [30, 30, 255],[90, 30, 150],[250, 80, 100],
                                    [180, 30, 80]], dtype=np.uint8)

    def get_weights(self):
        weights = torch.ones(self.config['nb_classes'])
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
        return len(self.all_files)

    def get_category(self, f_id):
        return str(self.all_files[f_id]).split("/")[-3]

    def get_object_name(self, f_id):
        return str(self.all_files[f_id]).split("/")[-1]

    def get_class_name(self, f_id):
        return "lidar"

    def get_save_dir(self, f_id):
        return os.path.join(str(self.all_files[f_id]).split("/")[-3], str(self.all_files[f_id]).split("/")[-2])



    def get(self, idx):
        """Get item."""

        fname_points = self.all_files[idx]
        frame_points = np.fromfile(fname_points, dtype=np.float32)
        pos = frame_points.reshape((-1, 4))
        intensities = pos[:,3:]
        pos = pos[:,:3]

        if self.split in ["test", "testing"]:
            # Fake labels
            y = np.zeros((pos.shape[0],), dtype=np.int32)
        else:
            # Read labels
            label_path = self.all_labels[idx]
            label = np.fromfile(label_path, dtype=np.uint32)
            label = label.reshape((-1))
            y = self.learning_map[label]

        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        intensities = torch.tensor(intensities, dtype=torch.float)
        x = torch.ones((pos.shape[0],1), dtype=torch.float)

        if "input_sem" in self.config and self.config["input_sem"]: 
            one_hot_semant = F.one_hot(y, num_classes=11)
            stacked = torch.hstack((x,one_hot_semant))
            x = stacked
            
        #Number of points in this scene
        N = torch.from_numpy(np.array(pos.shape[0]))[None]
        #Scene index
        shape_id = torch.from_numpy(np.array(idx))[None]

        return Data(x=x, intensities=intensities, pos=pos, y=y, 
                    shape_id=shape_id, N=N)