from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pandaset import DataSet, geometry
except: 
    print("!!!!!!!!!!!!!!!! ###########################")
    print(" NO PANDASET library available")
    print("!!!!!!!!!!!!!!!! ###########################")
import json
import os

import torch
from torch_geometric.data import Data, Dataset


def class_mapping_da(config):
    if config["source_dataset_name"] == "NuScenes" or  config["target_dataset_name"] == "NuScenes":
        # return {0:0,	#Noise --> Noise
        #     1:0,	#Smoke --> Noise
        #     2:0,	#Exhaust --> Noise
        #     3:0,	#Spray or rain --> Noise
        #     4:0,	#Reflection --> Noise
        #     5:15,	#Vegetation --> vegetation
        #     6:12,	#Ground --> Terrain/Other flat
        #     7:11,	#Road --> Driveable surface
        #     8:11,	#Lane Line Marking --> Driveable surface
        #     9:	11,	#Stop Line Marking --> Driveable surface
        #     10:	11,	#Other Road Marking --> Driveable surface
        #     11:	13,	#Sidewalk --> Sidewalk
        #     12:	11,	#Driveway --> Driveable surface
        #     13:	4,	#Car --> Car
        #     14:	10,	#Pickup Truck --> Truck
        #     15:	10,	#Medium-sized Truck --> Truck
        #     16:	10,	#Semi-truck --> Truck
        #     17:	9,	#Towed Object --> Trailer ? 
        #     18:	6,	#Motorcycle --> Motorcycle
        #     19:	5,	#Other Vehicle - Construction Vehicle --> Construction Vehicle
        #     20:	0,	#Other Vehicle - Uncommon --> Noise
        #     21:	0,	#Other Vehicle - Pedicab --> Noise
        #     22:	0,	#Emergency Vehicle --> Noise
        #     23:	3,	#Bus --> Bus
        #     24:	0,	#Personal Mobility Device --> Noise
        #     25:	0,	#Motorized Scooter --> Noise
        #     26:	2,	#Bicycle --> Bicycle
        #     27:	0,	#Train --> noise
        #     28:	0,	#Trolley --> noise 
        #     29:	0,	#Tram / Subway --> noise
        #     30:	7,	#Pedestrian --> Pedestrian
        #     31:	7,	#Pedestrian with Object  --> Pedestrian
        #     32:	0,	#Animals - Bird  --> 	Noise
        #     33:	0,	#Animals - Other --> 	Noise
        #     34:	8,	#Pylons --> 	(Traffic) Cone
        #     35:	1,	#Road Barriers --> Road Barriers
        #     36:	14,	#Signs --> Manmade
        #     37:	8,	#Cones --> (Traffic) Cone
        #     38:	14,	#Construction Signs --> Manmade
        #     39:	1,	#Temporary Construction Barriers --> Barriers
        #     40:	0,	#Rolling Containers --> Noise
        #     41:	14,	#Building --> Manmade
        #     42:	0}	#Other Static Object -->Noise
        return {0:	0,#	Noise -->	Noise
                1:	0,#	Smoke --> Noise
                2:	0,#	Exhaust --> Noise
                3:	0,#	Spray or rain --> Noise
                4:	0,#	Reflection --> Noise
                5:	7,#	Vegetation --> vegetation
                6:	5,#	Ground --> other ground
                7:	3,#	Road --> driveable ground
                8:	3,#	Lane Line Marking --> driveable ground
                9:	3,#	Stop Line Marking --> driveable ground
                10:	3,#	Other Road Marking --> driveable ground
                11:	4,#	Sidewalk --> sidewalk
                12:	3,#	Driveway --> driveable ground
                13:	8,#	Car --> 4-wheeled
                14:	8,#	Pickup Truck --> 4-wheeled
                15:	8,#	Medium-sized Truck --> 4-wheeled
                16:	8,#	Semi-truck --> 4-wheeled
                17:	8,#	Towed Object --> 4-wheeled
                18:	1,#	Motorcycle --> 2-wheeled
                19:	8,#	Other Vehicle - Construction Vehicle --> 4-wheeled
                20:	8,#	Other Vehicle - Uncommon --> 4-wheeled
                21:	1,#	Other Vehicle - Pedicab --> 2-wheeled
                22:	8,#	Emergency Vehicle --> 4-wheeled
                23:	8,#	Bus --> 4-wheeled
                24:	1,#	Personal Mobility Device --> 2-wheeled
                25:	1,#	Motorized Scooter --> 2-wheeled
                26:	1,#	Bicycle --> 2-wheeled
                27:	0,#	Train --> Noise
                28:	0,#	Trolley --> Noise
                29:	0,#	Tram / Subway --> Noise
                30:	2,#	Pedestrian --> pedestrian
                31:	2,#	Pedestrian with Object --> pedestrian
                32:	0,#	Animals - Bird --> Noise
                33:	0,#	Animals - Other --> Noise
                34:	6,#	Pylons --> manmade
                35:	6,#	Road Barriers --> manmade
                36:	6,#	Signs --> manmade
                37:	6,#	Cones --> manmade
                38:	6,#	Construction Signs --> manmade
                39:	6,#	Temporary Construction Barriers --> manmade
                40:	6,#	Rolling Containers --> manmade
                41:	6,#	Building --> manmade
                42:6}#	Other Static Object --> manmade
    
    elif config["source_dataset_name"] == "Pandaset" or  config["target_dataset_name"] == "Pandaset":
        return {0:	0,#	Noise -->	Noise
                1:	0,#	Smoke --> Noise
                2:	0,#	Exhaust --> Noise
                3:	0,#	Spray or rain --> Noise
                4:	0,#	Reflection --> Noise
                5:	7,#	Vegetation --> vegetation
                6:	5,#	Ground --> other ground
                7:	3,#	Road --> driveable ground
                8:	3,#	Lane Line Marking --> driveable ground
                9:	3,#	Stop Line Marking --> driveable ground
                10:	3,#	Other Road Marking --> driveable ground
                11:	4,#	Sidewalk --> sidewalk
                12:	3,#	Driveway --> driveable ground
                13:	8,#	Car --> 4-wheeled
                14:	8,#	Pickup Truck --> 4-wheeled
                15:	8,#	Medium-sized Truck --> 4-wheeled
                16:	8,#	Semi-truck --> 4-wheeled
                17:	8,#	Towed Object --> 4-wheeled
                18:	1,#	Motorcycle --> 2-wheeled
                19:	8,#	Other Vehicle - Construction Vehicle --> 4-wheeled
                20:	8,#	Other Vehicle - Uncommon --> 4-wheeled
                21:	1,#	Other Vehicle - Pedicab --> 2-wheeled
                22:	8,#	Emergency Vehicle --> 4-wheeled
                23:	8,#	Bus --> 4-wheeled
                24:	1,#	Personal Mobility Device --> 2-wheeled
                25:	1,#	Motorized Scooter --> 2-wheeled
                26:	1,#	Bicycle --> 2-wheeled
                27:	0,#	Train --> Noise
                28:	0,#	Trolley --> Noise
                29:	0,#	Tram / Subway --> Noise
                30:	2,#	Pedestrian --> pedestrian
                31:	2,#	Pedestrian with Object --> pedestrian
                32:	0,#	Animals - Bird --> Noise
                33:	0,#	Animals - Other --> Noise
                34:	6,#	Pylons --> manmade
                35:	6,#	Road Barriers --> manmade
                36:	6,#	Signs --> manmade
                37:	6,#	Cones --> manmade
                38:	6,#	Construction Signs --> manmade
                39:	6,#	Temporary Construction Barriers --> manmade
                40:	6,#	Rolling Containers --> manmade
                41:	6,#	Building --> manmade
                42:6}#	Other Static Object --> manmade

    else:
        raise ValueError("No mapping in Pandaset")

class Pandaset(Dataset):
    def __init__(self,root,split="training",transform=None, dataset_size=None,multiframe_range=None,
                 skip_ratio=1, da_flag =False, config=None, **kwargs):

        super().__init__(root, transform, None)
        dataset = DataSet(root)
        all_files_training = []
        all_poses_training = []
        all_labels_training = []


        all_files_test = []
        all_poses_test = []
        all_labels_test = []
        self.da_flag = da_flag
        self.use_ff_lidar = False #Use the Face Forward lidar (which is very dense)

        sequences = dataset.sequences(with_semseg=True)
        print("Number of detected sequences {}, {}".format(sequences, len(sequences)))
        for seq in sequences:
            _poses = []
            with open(Path(os.path.join(root, seq, "lidar/poses.json")), 'r') as f: 
                file_data = json.load(f)
                for entry in file_data:
                    _poses.append(entry)


            all_files_training += [x for x in  Path(os.path.join(root, seq, "lidar")).rglob('*.pkl.gz') if int(str(x).split("/")[-1].split(".")[0])<50 ] #The first 50 frames of a sequence
            all_labels_training += [x for x in  Path(os.path.join(root, seq, "annotations", "semseg")).rglob('*.pkl.gz') if int(str(x).split("/")[-1].split(".")[0])<50 ] #The first 50 frames of a sequence
            all_poses_training += _poses[0:50]

            all_files_test += [x for x in  Path(os.path.join(root, seq, "lidar")).rglob('*.pkl.gz') if int(str(x).split("/")[-1].split(".")[0])>=50 ] #The last 30 frames of a sequence
            all_labels_test += [x for x in  Path(os.path.join(root, seq, "annotations", "semseg")).rglob('*.pkl.gz') if int(str(x).split("/")[-1].split(".")[0])>=50 ] #The first 50 frames of a sequence
            all_poses_test += _poses[50:80]

        #Check that poses and lidar files are equal large, and 3800 frames for training as well as 2280 frames for test
        assert len(all_poses_training) == len(all_files_training) == len(all_labels_training) == 3800
        assert len(all_poses_test) == len(all_files_test) == len(all_labels_test) == 2280
        
        if split=='train': 
            self.all_files = all_files_training
            self.all_poses = all_poses_training
            self.all_labels = all_labels_training
        elif split=='val': 
            self.all_files = all_files_test
            self.all_poses = all_poses_test
            self.all_labels = all_labels_test
        else: 
            raise ValueError('Unknown set for Pandaset data: ', split)
            
        if self.da_flag:
            self.label_to_reduced = class_mapping_da(config) 

        self.label_to_reduced_np = np.zeros(43, dtype=np.int)
        for i in range(43):
            self.label_to_reduced_np[i] = self.label_to_reduced[i] 
            

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

    def __len__(self):
        return len(self.all_files)
    
    def len(self):
        return len(self.all_files)

    def get_category(self, f_id):
        pass

    def get_object_name(self, f_id):
        pass
    def get_class_name(self, f_id):
        pass
    
    def get_save_dir(self, f_id):
        pass

    def get_filename(self, idx):
        return self.all_files[idx]
    
    def get(self, idx):
        pos = pd.read_pickle(self.all_files[idx])
        label = pd.read_pickle(self.all_labels[idx])["class"].to_numpy()
        label = self.label_to_reduced_np[label]
        intensity = pos[["i"]].to_numpy()
        timestamp = pos[["t"]].to_numpy()
        sensor_flag = pos[["d"]].to_numpy() #0 is the mechanical sensor, 1 is the ff sensor
        pose = self.all_poses[idx]
        #Change from world coordinates to local corrdinates for the frame
        points3d_lidar_world = pos[["x", "y","z"]].to_numpy() #lidar_points
        points3d_lidar_local = geometry.lidar_points_to_ego(points3d_lidar_world, pose) 

        
        if self.use_ff_lidar:
            #Use forward facing lidar and mechanical lidar
            pass
        else:
            #Only use the mechanical lidar sensor
            mask = sensor_flag==0
            mask = mask [:,0]
            points3d_lidar_local = points3d_lidar_local[mask]
            label = label[mask]
            intensity = intensity[mask]/255 #normalize to 0 to 1
            timestamp = timestamp[mask]
            
        
        #Convert to torch
        pos = torch.tensor(points3d_lidar_local, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.long)
        intensities = torch.tensor(intensity, dtype=torch.float)
        x = torch.ones((pos.shape[0],1), dtype=torch.float)
        
        #Number of points in this scene
        N = torch.from_numpy(np.array(pos.shape[0]))[None]
        #Scene index
        shape_id = torch.from_numpy(np.array(idx))[None]
        
        return Data(x=x, intensities=intensities, pos=pos, y=y, shape_id=shape_id, N=N)
