import importlib
torchsparse_found = importlib.util.find_spec("torchsparse") is not None
import logging
logging.info(f"Torchsparse found - {torchsparse_found}")
if torchsparse_found:
    from torchsparse.utils.quantize import sparse_quantize
    from torchsparse import SparseTensor

import torch
import numpy as np


class TorchSparseQuantize(object):
    
    def __init__(self, voxel_size, item_list) -> None:
        self.voxel_size = voxel_size
        self.item_list = item_list

    def __call__(self, data):
        
        for item in self.item_list:
            
            pc_ = np.round(data[item].numpy() / self.voxel_size).astype(np.int32)
            pc_ -= pc_.min(0, keepdims=1)
            
            
            
                
            coords, indices, inverse_map = sparse_quantize(pc_,
                                                return_index=True,
                                                return_inverse=True)

            coords = torch.tensor(coords, dtype=torch.int)

            indices = torch.tensor(indices)
            feats = data["x"][indices]
            
            inverse_map = torch.tensor(inverse_map, dtype=torch.long)

            if item == "pos":
                data["sparse_input"] = SparseTensor(coords=coords, feats=feats)
                # data["sparse_input_invmap"] = SparseTensor(inverse_map, pc_)
                data["sparse_input_invmap"] = inverse_map
            elif item == "original_pos":
                data["original_sparse_input"] = SparseTensor(coords=coords, feats=feats)
                # data["sparse_input_invmap"] = SparseTensor(inverse_map, pc_)
                data["original_sparse_input_invmap"] = inverse_map
            else: 
                raise ValueError(f'Unknown item {item} in itemlist')
        return data


class MEQuantize(object):

    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __call__(self, data):

        pc_ = data["pos"].clone()
        pc_ /= self.voxel_size

        data["vox_pos"] = pc_

        return data