import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
import torch

__all__ = ['TorchSparseMinkUNet']


class New_BN_scaling_per_channel(torchsparse.nn.BatchNorm):
    def __init__(self, outc, learnable=True): 
        super().__init__(outc)
        self.ae_learned = torch.nn.Parameter(torch.ones((outc)))
        self.be_learned = torch.nn.Parameter(torch.zeros((1,outc)))
    
    def forward(self, x):
        x.F = x.F * self.ae_learned + self.be_learned
        x = super().forward(x)
        return x



class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, bn_func=spnn.BatchNorm):
        super().__init__()
        self.net_conv3d_0 = spnn.Conv3d(inc,outc,kernel_size=ks,dilation=dilation,stride=stride)
        self.net_bn_1  =  bn_func(outc)
        self.relu = spnn.ReLU(True)
        

    def forward(self, x):
        return_dict={}
        x0 = self.net_conv3d_0(x)
        x0 = self.net_bn_1(x0)
        out = self.relu(x0)
        return out, return_dict



class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, bn_func=spnn.BatchNorm):
        super().__init__()
        self.net_conv3d_0 = spnn.Conv3d(inc, outc, kernel_size=ks, stride=stride, transposed=True)
        self.net_bn_1 = bn_func(outc)
        self.relu = spnn.ReLU(True)
        

    def forward(self, x):
        return_dict={}
        x0 = self.net_conv3d_0(x)
        x0 = self.net_bn_1(x0)
        out = self.relu(x0)
        
        return out, return_dict
    

class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, bn_func=spnn.BatchNorm):
        super().__init__()
        self.net_conv3d_0 = spnn.Conv3d(inc,outc,kernel_size=ks,dilation=dilation,stride=stride)
        self.net_bn_1 = bn_func(outc)
        #spnn.ReLU(True),
        self.net_conv3d_2 = spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,stride=1)
        self.net_bn_3 = bn_func(outc)
        

        if inc == outc and stride == 1:
            self.downsample_flag=False
            self.downsample = nn.Sequential()
        else:
            self.downsample_flag=True
            self.downsample_conv3d_0 = spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,stride=stride)
            self.downsample_bn_1 =  bn_func(outc)
            

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        return_dict = {}
        x0 = self.net_conv3d_0(x)
        x0 = self.net_bn_1(x0)
        x0 = self.relu(x0)
        x0 = self.net_conv3d_2(x0)
        x0 = self.net_bn_3(x0)
        
        if self.downsample_flag:
            x_ds = self.downsample_conv3d_0(x)
            x_ds = self.downsample_bn_1(x_ds)
        else:
            x_ds = x
        
        out = self.relu(x0 + x_ds)
        return out, return_dict


class MinkUNet_learned(nn.Module):
    # Gives access to intermediate values

    def __init__(self, **kwargs):
        super().__init__()

        self.bn_layer_name = kwargs["config"]["parameter"]["bn_layer"]
        
        
        bn_func = None

        if self.bn_layer_name == "standard": 
            bn_func = spnn.BatchNorm
        elif self.bn_layer_name == "scaling_per_channel":
            bn_func = New_BN_scaling_per_channel
        else: 
            raise NotImplementedError      
        
        bn_func_standard = bn_func
        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)
        self.relu = spnn.ReLU(True)

        in_channels = kwargs["in_channels"]

        self.stem_conv0 = spnn.Conv3d(in_channels, cs[0], kernel_size=3, stride=1)
        self.stem_bn0 = bn_func(cs[0])
        #spnn.ReLU(True),
        self.stem_conv1 = spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1)
        self.stem_bn1 = bn_func(cs[0]) 
        #spnn.ReLU(True)

        self.stage1_BC_0 = BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1, bn_func=bn_func_standard)
        self.stage1_RB_1 = ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)
        self.stage1_RB_2 = ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)
        

        self.stage2_BC_0 = BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1, bn_func=bn_func_standard)
        self.stage2_RB_1 = ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)
        self.stage2_RB_2 = ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)

        self.stage3_BC_0 = BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1, bn_func=bn_func_standard)
        self.stage3_RB_1 = ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)
        self.stage3_RB_2 = ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)

        self.stage4_BC_0 = BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1, bn_func=bn_func_standard)
        self.stage4_RB_1 = ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)
        self.stage4_RB_2 = ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)
        

        self.up1_BC_0 = BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2, bn_func=bn_func_standard)
        self.up1_2_RB_1 = ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)
        self.up1_2_RB_2 = ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)
                

        self.up2_BC_0 = BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2, bn_func=bn_func_standard)
        self.up2_2_RB_1 = ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)
        self.up2_2_RB_2 = ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)

        self.up3_BC_0 = BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2, bn_func=bn_func_standard)
        self.up3_2_RB_1 = ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)
        self.up3_2_RB_2 = ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)
       

        self.up4_BC_0 = BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2, bn_func=bn_func_standard)
        self.up4_2_RB_1 = ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)
        self.up4_2_RB_2 =  ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1, bn_func=bn_func_standard)
        

        self.classifier = nn.Sequential(nn.Linear(cs[8], kwargs['out_channels'])) 


        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)
        
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem_conv0(x)
        x0 = self.stem_bn0(x0)

        x0= self.relu(x0)
        x0=self.stem_conv1(x0)
        x0=self.stem_bn1(x0)
        x0=self.relu(x0)

        return_dict = {"conv0_f":None, "conv1_f":None}
        # Stage1
        x1, stage1_bc_dict = self.stage1_BC_0(x0) 
        x1, stage1_rb_1_dict = self.stage1_RB_1(x1)
        x1, stage1_rb_2_dict = self.stage1_RB_2(x1) 

        
        
        # Stage2
        x2, stage2_bc_dict = self.stage2_BC_0(x1) 
        x2, stage2_rb_1_dict = self.stage2_RB_1(x2) 
        x2, stage2_rb_2_dict = self.stage2_RB_2(x2)

        # Stage3
        x3, stage3_bc_dict = self.stage3_BC_0(x2)
        x3, stage3_rb_1_dict = self.stage3_RB_1(x3)
        x3, stage3_rb_2_dict = self.stage3_RB_2(x3)

        # Stage4
        x4, stage4_bc_dict = self.stage4_BC_0(x3)
        x4, stage4_rb_1_dict = self.stage4_RB_1(x4)
        x4, stage4_rb_2_dict = self.stage4_RB_2(x4)

        # Up1
        y1, up1_bc_dict = self.up1_BC_0(x4)
        y1 = torchsparse.cat([y1, x3])
        y1, up1_2_rb_1_dict = self.up1_2_RB_1(y1)
        y1, up1_2_rb_2_dict = self.up1_2_RB_2(y1)

        # Up2
        y2, up2_bc_dict = self.up2_BC_0(y1)
        y2 = torchsparse.cat([y2, x2])
        y2, up2_2_rb_1_dict = self.up2_2_RB_1(y2)
        y2, up2_2_rb_2_dict = self.up2_2_RB_2(y2)

        # Up3
        y3, up3_bc_dict = self.up3_BC_0(y2)
        y3 = torchsparse.cat([y3, x1])
        y3, up3_2_rb_1_dict = self.up3_2_RB_1(y3)
        y3, up3_2_rb_2_dict = self.up3_2_RB_2(y3)

        # Up4
        y4, up4_bc_dict = self.up4_BC_0(y3)
        y4 = torchsparse.cat([y4, x0])
        y4, up4_2_rb_1_dict = self.up4_2_RB_1(y4)
        y4, up4_2_rb_2_dict = self.up4_2_RB_2(y4)

        

        out = self.classifier(y4.F)

        return out

    
    

    def forward_mapped_learned(self, x):
        return_dict = {}
    
        x0 = self.stem_conv0(x)
        x0 = self.stem_bn0(x0)
        x0= self.relu(x0)
        x0=self.stem_conv1(x0)
        x0=self.stem_bn1(x0)
        x0=self.relu(x0)
       

        # Stage1
        x1, stage1_bc_dict = self.stage1_BC_0(x0)
        x1, stage1_rb_1_dict = self.stage1_RB_1(x1)
        x1, stage1_rb_2_dict = self.stage1_RB_2(x1) 
        
        # Stage2
        x2, stage2_bc_dict = self.stage2_BC_0(x1) 
        x2, stage2_rb_1_dict = self.stage2_RB_1(x2)
        x2, stage2_rb_2_dict = self.stage2_RB_2(x2)
        
        # Stage3
        x3, stage3_bc_dict = self.stage3_BC_0(x2)
        x3, stage3_rb_1_dict = self.stage3_RB_1(x3)
        x3, stage3_rb_2_dict = self.stage3_RB_2(x3)

        # Stage4
        x4, stage4_bc_dict = self.stage4_BC_0(x3)
        x4, stage4_rb_1_dict = self.stage4_RB_1(x4)
        x4, stage4_rb_2_dict = self.stage4_RB_2(x4)

        # Up1
        y1, up1_bc_dict = self.up1_BC_0(x4)
        y1 = torchsparse.cat([y1, x3])
        y1, up1_2_rb_1_dict = self.up1_2_RB_1(y1)
        y1, up1_2_rb_2_dict = self.up1_2_RB_2(y1)

        # Up2
        y2, up2_bc_dict = self.up2_BC_0(y1)
        y2 = torchsparse.cat([y2, x2])
        y2, up2_2_rb_1_dict = self.up2_2_RB_1(y2)
        y2, up2_2_rb_2_dict = self.up2_2_RB_2(y2)

        # Up3
        y3, up3_bc_dict = self.up3_BC_0(y2)
        y3 = torchsparse.cat([y3, x1])
        y3, up3_2_rb_1_dict = self.up3_2_RB_1(y3)
        y3, up3_2_rb_2_dict = self.up3_2_RB_2(y3)

        # Up4
        y4, up4_bc_dict = self.up4_BC_0(y3)
        y4 = torchsparse.cat([y4, x0])
        y4, up4_2_rb_1_dict = self.up4_2_RB_1(y4)
        y4, up4_2_rb_2_dict = self.up4_2_RB_2(y4)

        
        out = self.classifier(y4.F)
        
        return out, return_dict



class TorchSparseMinkUNet_learned(MinkUNet_learned):

    def __init__(self, in_channels, out_channels,
                voxel_size=1,
                cylindrical_coordinates=False,
                cr=1.0,
                **kwargs
                ):
        
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels, cr=cr, config=kwargs["config"]) ####"num classes is here used

        self.voxel_size = voxel_size
        self.cylindrical_coordinates = cylindrical_coordinates

    def forward_spatial(self, data):
        return data

    def get_stack_item_list(self):
        return []

    def get_cat_item_list(self):
        return []

    def forward(self, data):

        # forward in the network
        outputs = super().forward(data["sparse_input"])

        # interpolate the outputs
        outputs = outputs[data["sparse_input_invmap"]]

        return outputs

    
    def forward_mapped_learned(self, data):

        # forward in the network
        outputs, x0 = super().forward_mapped_learned(data["sparse_input"])

        # interpolate the outputs
        outputs = outputs[data["sparse_input_invmap"]]

        return outputs, x0
    
    def forward_mapped_learned_original(self, data): 
        # In the case the original_sparse_input is used
        
        # forward in the network
        outputs, x0 = super().forward_mapped_learned(data["original_sparse_input"])

        # interpolate the outputs
        outputs = outputs[data["original_sparse_input_invmap"]]

        return outputs, x0

    
    

    @staticmethod
    def get_final_layer_name():
        return "classifier"


    @staticmethod
    def get_linear_layer(in_channels, out_channels):
        return nn.Conv1d(in_channels, out_channels, 1)
