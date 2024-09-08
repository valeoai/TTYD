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
from os import listdir
from os.path import exists, isdir, join
from pathlib import Path

# Basic libs
import numpy as np
import yaml
from scipy.spatial import cKDTree

# Define PLY types
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}

def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None


    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)

    return num_points, num_faces, vertex_properties

def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to read.
    Returns
    -------
    result : array
        data stored in the file
    Examples
    --------
    Store data in file
    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])
    Read the file
    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])
    
    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """

    with open(filename, 'rb') as plyfile:


        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [('k', ext + 'u1'),
                               ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'),
                               ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


class SemanticKITTI(Dataset):

    

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
        self.n_frames = 1
        self.da_flag = da_flag
        self.config = config
        self.multiframe_range = multiframe_range
        self.N_LABELS = self.config["nb_classes"] if self.config is not None else 11

        
        logging.info(f"SemanticKITTI - split {split}")

        # get the scenes
        assert(split in ["train", "val", "test", "verifying", "parametrizing", "ssda"])
        if split == "verifying":
            self.sequences = ['{:02d}'.format(i) for i in range(11) if i == 7]
        elif split == "parametrizing":
            self.sequences = ['{:02d}'.format(i) for i in range(11) if (i != 8 and i != 7)]
        elif split == "train":
            self.sequences = ['{:02d}'.format(i) for i in range(11) if i != 8]
        elif split == "val":
            self.sequences = ['{:02d}'.format(i) for i in range(11) if i == 8]
        elif split == "ssda": 
            #Split for semi supervised setting
            self.sequences = ['{:02d}'.format(i) for i in range(11) if ((i == 2) or (i == 6))]
        elif split == "test":
            self.sequences = ['{:02d}'.format(i) for i in range(11, 22)]
        else:
            raise ValueError('Unknown set for SemanticKitti data: ', split)

        # get the filenames
        self.all_files = []
        for sequence in self.sequences:
            self.all_files += [path for path in Path(os.path.join(self.root, "dataset", "sequences", sequence, "velodyne")).rglob('*.bin')]
        
        #Filter for ssda
        if split == "ssda": 
            #  848 from sequence 06 
            #  940 from sequence 02
            self.all_files = []
            self.all_files += [Path(os.path.join(self.root, "dataset", "sequences", "06", "velodyne",'000848.bin'))]
            self.all_files += [Path(os.path.join(self.root, "dataset", "sequences", "02", "velodyne",'000940.bin'))]
        
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

        with open(config_file, 'r') as stream:
            doc = yaml.safe_load(stream)
            all_labels = doc['labels']
            
            if self.da_flag: 
                #Changes of mapping in DA case
                if self.config["source_dataset_name"] == "NuScenes" or  self.config["target_dataset_name"] == "NuScenes": 
                    learning_map={
                        0 : 0,     # "unlabeled"
                        1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
                        10: 1,     # "car"
                        11: 2,    # "bicycle"
                        13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
                        15: 3,     # "motorcycle"
                        16: 0,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
                        18: 4,     # "truck"
                        20: 5,     # "other-vehicle"
                        30: 6,     # "person"
                        31: 0,     # "bicyclist"
                        32: 0,     # "motorcyclist"
                        40: 7,     # "road"
                        44: 7,    # "parking"
                        48: 8,    # "sidewalk"
                        49: 0,    # "other-ground"
                        50: 0,    # "building"
                        51: 0,    # "fence"
                        52: 0,    # "other-structure" mapped to "unlabeled" ------------------mapped
                        60: 7,     # "lane-marking" to "road" ---------------------------------mapped
                        70: 10,    # "vegetation"
                        71: 10,    # "trunk"
                        72: 9,    # "terrain"
                        80: 0,    # "pole"
                        81: 0,    # "traffic-sign"
                        99: 0,    # "other-object" to "unlabeled" ----------------------------mapped
                        252: 1,    # "moving-car" to "car" ------------------------------------mapped
                        253: 0,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
                        254: 6,    # "moving-person" to "person" ------------------------------mapped
                        255: 0,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
                        256: 0,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
                        257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
                        258: 4,    # "moving-truck" to "truck" --------------------------------mapped
                        259: 5    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
                    }
                    learning_map_inv ={ # inverse of previous map
                        0: 0,      # "unlabeled", and others ignored
                        1: 10,     # "car"
                        2: 11,     # "bicycle"
                        3: 15,     # "motorcycle"
                        4: 18,     # "truck"
                        5: 20,     # "other-vehicle"
                        6: 30,     # "person"
                        #: 31,     # "bicyclist" No differentitation to bicycle
                        #8: 32,     # "motorcyclist" No differentiation to motorcycle
                        7: 40,     # "road"
                        #: 44,    # "parking" No differentation to road
                        8: 48,    # "sidewalk"
                        #: 49,    # "other-ground" Ignored
                        #: 50,    # "building"
                        #: 51,    # "fence"
                        10: 70,    # "vegetation"
                        #: 71,    # "trunk"is in vegetation
                        9: 72,    # "terrain"
                        #: 80,    # "pole"
                        #: 81,    # "traffic-sign"
                    
                    }
                elif self.config["source_dataset_name"] == "SynLidar" or  self.config["target_dataset_name"] == "SynLidar": 
                    #Mapping with SynLiDAR
                    learning_map = doc['learning_map']
                    learning_map_inv = doc['learning_map_inv']

                elif self.config["source_dataset_name"] == "Livox" or  self.config["target_dataset_name"] == "Livox":
                        learning_map={
                            0 : 0,     # "unlabeled"
                        1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
                        10: 1,     # "car"
                        11: 4,    # "bicycle"
                        13: 3,     # "bus" mapped to "other-vehicle" --------------------------mapped
                        15: 5,     # "motorcycle"
                        16: 0,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
                        18: 2,     # "truck"
                        20: 0,     # "other-vehicle"
                        30: 6,     # "person"
                        31: 4,     # "bicyclist"
                        32: 5,     # "motorcyclist"
                        40: 7,     # "road"
                        44: 7,    # "parking"
                        48: 0,    # "sidewalk"
                        49: 0,    # "other-ground"
                        50: 8,    # "building"
                        51: 9,    # "fence"
                        52: 0,    # "other-structure" mapped to "unlabeled" ------------------mapped
                        60: 7,     # "lane-marking" to "road" ---------------------------------mapped
                        70: 0,    # "vegetation"
                        71: 0,    # "trunk"
                        72: 0,    # "terrain"
                        80: 10,    # "pole"
                        81: 0,    # "traffic-sign"
                        99: 0,    # "other-object" to "unlabeled" ----------------------------mapped
                        252: 1,    # "moving-car" to "car" ------------------------------------mapped
                        253: 4,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
                        254: 6,    # "moving-person" to "person" ------------------------------mapped
                        255: 5,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
                        256: 0,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
                        257: 3,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
                        258: 2,    # "moving-truck" to "truck" --------------------------------mapped
                        259: 0    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
                        }

                        learning_map_inv={
                            0 : 0,     # "unlabeled"
                                    # "outlier" mapped to "unlabeled" --------------------------mapped
                        1: 10,     # "car"
                        2:18,     # "truck"
                        3: 13,     # "bus" mapped to "other-vehicle" --------------------------mapped
                        4: 11,    # "bicycle"
                        5: 15,     # "motorcycle"
                        6:30,     # "person"
                        7:40,     # "road"
                        8:50,    # "building"
                        9:51,    # "fence"
                        10: 80,    # "pole"
                        }
                elif self.config["source_dataset_name"] == "SemanticKITTI" or  self.config["target_dataset_name"] == "SemanticKITTI":
                    learning_map = doc['learning_map']
                    learning_map_inv = doc['learning_map_inv']
                else:
                    ValueError("No source/target mapping available")
            else:
                ValueError("No source/target mapping available")
            
            self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
            
            for k, v in learning_map.items():
                self.learning_map[k] = v
            
            self.learning_map_inv = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map_inv.items():
                self.learning_map_inv[k] = v
          
        
        #Color in BGR
        # self.class_colors = {
        #     0 : [0, 0, 0],
        #     1: [245, 150, 100],
        #     2: [245, 230, 100],
        #     3: [150, 60, 30],
        #     4: [180, 30, 80],
        #     5: [255, 0, 0],
        #     6: [30, 30, 255],
        #     7: [200, 40, 255],
        #     8: [90, 30, 150],
        #     9: [255, 0, 255],
        #     10: [255, 150, 255],
        #     11: [75, 0, 75],
        #     12: [75, 0, 175],
        #     13: [0, 200, 255],
        #     14: [50, 120, 255],
        #     15: [0, 175, 0],
        #     16: [0, 60, 135],
        #     17: [80, 240, 150],
        #     18: [150, 240, 255],
        #     19: [0, 0, 255],
        # }

        self.class_colors = np.array([
            [0, 0, 0],
            [245, 150, 100],
            [245, 230, 100],
            [150, 60, 30],
            [180, 30, 80],
            [255, 0, 0],
            [30, 30, 255],
            [200, 40, 255],
            [90, 30, 150],
            [255, 0, 255],
            [255, 150, 255],
            [75, 0, 75],
            [75, 0, 175],
            [0, 200, 255],
            [50, 120, 255],
            [0, 175, 0],
            [0, 60, 135],
            [80, 240, 150],
            [150, 240, 255],
            [0, 0, 255],
        ], dtype=np.uint8)

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
        return len(self.all_files)

    def get_category(self, f_id):
        return str(self.all_files[f_id]).split("/")[-3]

    def get_object_name(self, f_id):
        return str(self.all_files[f_id]).split("/")[-1]

    def get_class_name(self, f_id):
        return "lidar"

    def get_save_dir(self, f_id):
        return os.path.join(str(self.all_files[f_id]).split("/")[-3], str(self.all_files[f_id]).split("/")[-2])

    def get_filename(self, idx):
        return self.all_files[idx]

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
            label_file = self.all_labels[idx]
            frame_labels = np.fromfile(label_file, dtype=np.int32)
            y = frame_labels & 0xFFFF  # semantic label in lower half
            y = self.learning_map[y]

        # points are annotated only until 50 m

        mask = np.linalg.norm(pos, axis=1)<50
        pos = pos[mask]
        y = y[mask]
        intensities = intensities[mask]

       


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
        return Data(x=x, intensities=intensities, pos=pos, y=y, shape_id=shape_id, N=N)

