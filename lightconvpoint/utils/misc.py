import importlib

import torch

torchsparse_found = importlib.util.find_spec("torchsparse") is not None
if torchsparse_found:
    from torchsparse import SparseTensor


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def dict_to_device(data, device):
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to(device)
        elif isinstance(value, list):
            data[key] = list_to_device(value, device)
        elif isinstance(value, dict):
            data[key] = dict_to_device(value, device)
        elif isinstance(value, SparseTensor):
            data[key] = data[key].to(device)
    return data
