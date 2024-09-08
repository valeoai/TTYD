import logging
import torch
import torch.nn as nn

from .backbone import *



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Network(torch.nn.Module):

    def __init__(self, in_channels, latent_size, backbone,
                voxel_size = 1, dual_seg_head = False,
                target_in_channels = None,
                config=None,
                **kwargs):
        super().__init__()

        self.backbone = backbone
        self.dual_seg_head_flag = dual_seg_head
        self.config = config
       
        self.latent_size = latent_size
        self.activation = nn.ReLU(inplace=True)
        self.net = get_backbone(backbone)(in_channels, self.latent_size, 
                            segmentation=True, dropout=0, spatial_prefix="encoder_",
                            voxel_size = voxel_size, nb_classes=self.config["nb_classes"], 
                            target_in_channels=target_in_channels, config=config)
        

        
        logging.info(f"Network -- backbone -- {count_parameters(self.net)} parameters")

        
        print("Latent size creation {} and nb classes creation {}, larger seg head: False ".format(self.latent_size, self.config["nb_classes"]))
        self.dual_seg_head = self.net.get_linear_layer(self.latent_size, self.config["nb_classes"])
        logging.info(f"Network -- Seg. head -- {count_parameters(self.dual_seg_head)} parameters")

            


    def train(self, mode=True):
        r"""Sets the module in training mode."""      
        self.training = mode
        self.net.train(mode)
        return self


    def forward_spatial(self, data):
        data = self.net.forward_spatial(data)
        return data
    
    def get_stack_item_list(self):
        item_list = self.net.get_stack_item_list()
        return item_list

    def get_cat_item_list(self):
        item_list = self.net.get_cat_item_list()
        return item_list

        

    def forward_pretraining_inside(self, data, return_latents=False, return_projection=True, get_latent=False, inference=False):
        data["latents"] = self.net(data)
        
        outs_sem = self.dual_seg_head(data["latents"][:,:,None])
        if get_latent: 
            return data["latents"], outs_sem, None
        else:
            return None, outs_sem, None
    
    
    def forward_mapped_learned(self, data, inference=False):
        data["latents"], return_dict= self.net.forward_mapped_learned(data)
        outs_sem = self.dual_seg_head(data["latents"][:,:,None])
        return None, outs_sem, return_dict
            
    def forward_mapped_learned_original(self, data, inference=False):
        data["latents"], return_dict= self.net.forward_mapped_learned_original(data)
        outs_sem = self.dual_seg_head(data["latents"][:,:,None])
        return None, outs_sem, return_dict
        