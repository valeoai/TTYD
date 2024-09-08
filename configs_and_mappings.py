import torch
import yaml
import numpy as np


name_shift = {"net.stem_conv0.kernel":"net.stem.0.kernel",
"net.stem_bn0.weight":"net.stem.1.weight",
"net.stem_bn0.bias":"net.stem.1.bias",
"net.stem_bn0.running_mean":"net.stem.1.running_mean",
"net.stem_bn0.running_var":"net.stem.1.running_var",
"net.stem_conv1.kernel":"net.stem.3.kernel",
"net.stem_bn1.weight":"net.stem.4.weight",
"net.stem_bn1.bias":"net.stem.4.bias",
"net.stem_bn1.running_mean":"net.stem.4.running_mean",
"net.stem_bn1.running_var":"net.stem.4.running_var",
"net.stage1_BC_0.net_conv3d_0.kernel":"net.stage1.0.net.0.kernel",
"net.stage1_BC_0.net_bn_1.weight":"net.stage1.0.net.1.weight",
"net.stage1_BC_0.net_bn_1.bias":"net.stage1.0.net.1.bias",
"net.stage1_BC_0.net_bn_1.running_mean":"net.stage1.0.net.1.running_mean",
"net.stage1_BC_0.net_bn_1.running_var":"net.stage1.0.net.1.running_var",
"net.stage1_RB_1.net_conv3d_0.kernel": "net.stage1.1.net.0.kernel",
"net.stage1_RB_1.net_bn_1.weight": "net.stage1.1.net.1.weight",
"net.stage1_RB_1.net_bn_1.bias": "net.stage1.1.net.1.bias",
"net.stage1_RB_1.net_bn_1.running_mean":"net.stage1.1.net.1.running_mean", 
"net.stage1_RB_1.net_bn_1.running_var":"net.stage1.1.net.1.running_var", 
"net.stage1_RB_1.net_conv3d_2.kernel":"net.stage1.1.net.3.kernel", 
"net.stage1_RB_1.net_bn_3.weight":"net.stage1.1.net.4.weight", 
"net.stage1_RB_1.net_bn_3.bias":"net.stage1.1.net.4.bias", 
"net.stage1_RB_1.net_bn_3.running_mean":"net.stage1.1.net.4.running_mean",
"net.stage1_RB_1.net_bn_3.running_var":"net.stage1.1.net.4.running_var",
"net.stage1_RB_2.net_conv3d_0.kernel":"net.stage1.2.net.0.kernel",
"net.stage1_RB_2.net_bn_1.weight":"net.stage1.2.net.1.weight",
"net.stage1_RB_2.net_bn_1.bias":"net.stage1.2.net.1.bias",
"net.stage1_RB_2.net_bn_1.running_mean":"net.stage1.2.net.1.running_mean",
"net.stage1_RB_2.net_bn_1.running_var":"net.stage1.2.net.1.running_var",
"net.stage1_RB_2.net_conv3d_2.kernel":"net.stage1.2.net.3.kernel",
"net.stage1_RB_2.net_bn_3.weight":"net.stage1.2.net.4.weight",
"net.stage1_RB_2.net_bn_3.bias":"net.stage1.2.net.4.bias",
"net.stage1_RB_2.net_bn_3.running_mean":"net.stage1.2.net.4.running_mean",
"net.stage1_RB_2.net_bn_3.running_var":"net.stage1.2.net.4.running_var",

"net.stage2_BC_0.net_conv3d_0.kernel": "net.stage2.0.net.0.kernel",
"net.stage2_BC_0.net_bn_1.weight":"net.stage2.0.net.1.weight",
"net.stage2_BC_0.net_bn_1.bias":"net.stage2.0.net.1.bias",
"net.stage2_BC_0.net_bn_1.running_mean":"net.stage2.0.net.1.running_mean",
"net.stage2_BC_0.net_bn_1.running_var":"net.stage2.0.net.1.running_var",
"net.stage2_RB_1.net_conv3d_0.kernel":"net.stage2.1.net.0.kernel",
"net.stage2_RB_1.net_bn_1.weight":"net.stage2.1.net.1.weight",
"net.stage2_RB_1.net_bn_1.bias":"net.stage2.1.net.1.bias",
"net.stage2_RB_1.net_bn_1.running_mean":"net.stage2.1.net.1.running_mean",
"net.stage2_RB_1.net_bn_1.running_var":"net.stage2.1.net.1.running_var",
"net.stage2_RB_1.net_conv3d_2.kernel":"net.stage2.1.net.3.kernel",
"net.stage2_RB_1.net_bn_3.weight":"net.stage2.1.net.4.weight",
"net.stage2_RB_1.net_bn_3.bias":"net.stage2.1.net.4.bias",
"net.stage2_RB_1.net_bn_3.running_mean":"net.stage2.1.net.4.running_mean",
"net.stage2_RB_1.net_bn_3.running_var":"net.stage2.1.net.4.running_var",
"net.stage2_RB_1.downsample_conv3d_0.kernel":"net.stage2.1.downsample.0.kernel",
"net.stage2_RB_1.downsample_bn_1.weight":"net.stage2.1.downsample.1.weight",
"net.stage2_RB_1.downsample_bn_1.bias":"net.stage2.1.downsample.1.bias",
"net.stage2_RB_1.downsample_bn_1.running_mean":"net.stage2.1.downsample.1.running_mean",
"net.stage2_RB_1.downsample_bn_1.running_var":"net.stage2.1.downsample.1.running_var",
"net.stage2_RB_2.net_conv3d_0.kernel":"net.stage2.2.net.0.kernel",
"net.stage2_RB_2.net_bn_1.weight":"net.stage2.2.net.1.weight",
"net.stage2_RB_2.net_bn_1.bias": "net.stage2.2.net.1.bias",

"net.stage2_RB_2.net_bn_1.running_mean":"net.stage2.2.net.1.running_mean",
"net.stage2_RB_2.net_bn_1.running_var":"net.stage2.2.net.1.running_var",
"net.stage2_RB_2.net_conv3d_2.kernel":"net.stage2.2.net.3.kernel",
"net.stage2_RB_2.net_bn_3.weight":"net.stage2.2.net.4.weight",
"net.stage2_RB_2.net_bn_3.bias":"net.stage2.2.net.4.bias",
"net.stage2_RB_2.net_bn_3.running_mean":"net.stage2.2.net.4.running_mean",
"net.stage2_RB_2.net_bn_3.running_var":"net.stage2.2.net.4.running_var",
"net.stage3_BC_0.net_conv3d_0.kernel":"net.stage3.0.net.0.kernel",
"net.stage3_BC_0.net_bn_1.weight":"net.stage3.0.net.1.weight",
"net.stage3_BC_0.net_bn_1.bias":"net.stage3.0.net.1.bias",
"net.stage3_BC_0.net_bn_1.running_mean":"net.stage3.0.net.1.running_mean",
"net.stage3_BC_0.net_bn_1.running_var":"net.stage3.0.net.1.running_var",
"net.stage3_RB_1.net_conv3d_0.kernel":"net.stage3.1.net.0.kernel",
"net.stage3_RB_1.net_bn_1.weight":"net.stage3.1.net.1.weight",
"net.stage3_RB_1.net_bn_1.bias":"net.stage3.1.net.1.bias",
"net.stage3_RB_1.net_bn_1.running_mean":"net.stage3.1.net.1.running_mean",
"net.stage3_RB_1.net_bn_1.running_var":"net.stage3.1.net.1.running_var",
"net.stage3_RB_1.net_conv3d_2.kernel":"net.stage3.1.net.3.kernel", 
"net.stage3_RB_1.net_bn_3.weight":"net.stage3.1.net.4.weight",
"net.stage3_RB_1.net_bn_3.bias":"net.stage3.1.net.4.bias",
"net.stage3_RB_1.net_bn_3.running_mean":"net.stage3.1.net.4.running_mean",
"net.stage3_RB_1.net_bn_3.running_var":"net.stage3.1.net.4.running_var",
"net.stage3_RB_1.downsample_conv3d_0.kernel":"net.stage3.1.downsample.0.kernel",
"net.stage3_RB_1.downsample_bn_1.weight":"net.stage3.1.downsample.1.weight",
"net.stage3_RB_1.downsample_bn_1.bias":"net.stage3.1.downsample.1.bias",
"net.stage3_RB_1.downsample_bn_1.running_mean":"net.stage3.1.downsample.1.running_mean",
"net.stage3_RB_1.downsample_bn_1.running_var":"net.stage3.1.downsample.1.running_var",
"net.stage3_RB_2.net_conv3d_0.kernel":"net.stage3.2.net.0.kernel",
"net.stage3_RB_2.net_bn_1.weight":"net.stage3.2.net.1.weight",
"net.stage3_RB_2.net_bn_1.bias":"net.stage3.2.net.1.bias",
"net.stage3_RB_2.net_bn_1.running_mean":"net.stage3.2.net.1.running_mean",
"net.stage3_RB_2.net_bn_1.running_var":"net.stage3.2.net.1.running_var",
"net.stage3_RB_2.net_conv3d_2.kernel":"net.stage3.2.net.3.kernel",
"net.stage3_RB_2.net_bn_3.weight":"net.stage3.2.net.4.weight",
"net.stage3_RB_2.net_bn_3.bias":"net.stage3.2.net.4.bias",
"net.stage3_RB_2.net_bn_3.running_mean":"net.stage3.2.net.4.running_mean",
"net.stage3_RB_2.net_bn_3.running_var":"net.stage3.2.net.4.running_var",

"net.stage4_BC_0.net_conv3d_0.kernel":"net.stage4.0.net.0.kernel",
"net.stage4_BC_0.net_bn_1.weight": "net.stage4.0.net.1.weight", 
"net.stage4_BC_0.net_bn_1.bias":"net.stage4.0.net.1.bias",
"net.stage4_BC_0.net_bn_1.running_mean":"net.stage4.0.net.1.running_mean", 
"net.stage4_BC_0.net_bn_1.running_var":"net.stage4.0.net.1.running_var", 
"net.stage4_RB_1.net_conv3d_0.kernel":"net.stage4.1.net.0.kernel", 
"net.stage4_RB_1.net_bn_1.weight":"net.stage4.1.net.1.weight", 
"net.stage4_RB_1.net_bn_1.bias":"net.stage4.1.net.1.bias", 
"net.stage4_RB_1.net_bn_1.running_mean":"net.stage4.1.net.1.running_mean", 
"net.stage4_RB_1.net_bn_1.running_var":"net.stage4.1.net.1.running_var", 
"net.stage4_RB_1.net_conv3d_2.kernel":"net.stage4.1.net.3.kernel", 
"net.stage4_RB_1.net_bn_3.weight":"net.stage4.1.net.4.weight", 
"net.stage4_RB_1.net_bn_3.bias":"net.stage4.1.net.4.bias", 
"net.stage4_RB_1.net_bn_3.running_mean":"net.stage4.1.net.4.running_mean", 
"net.stage4_RB_1.net_bn_3.running_var":"net.stage4.1.net.4.running_var", 
"net.stage4_RB_1.downsample_conv3d_0.kernel": "net.stage4.1.downsample.0.kernel", 
"net.stage4_RB_1.downsample_bn_1.weight":"net.stage4.1.downsample.1.weight",
"net.stage4_RB_1.downsample_bn_1.bias":"net.stage4.1.downsample.1.bias", 
"net.stage4_RB_1.downsample_bn_1.running_mean":"net.stage4.1.downsample.1.running_mean", 
"net.stage4_RB_1.downsample_bn_1.running_var":"net.stage4.1.downsample.1.running_var", 
"net.stage4_RB_2.net_conv3d_0.kernel":"net.stage4.2.net.0.kernel", 
"net.stage4_RB_2.net_bn_1.weight":"net.stage4.2.net.1.weight", 
"net.stage4_RB_2.net_bn_1.bias":"net.stage4.2.net.1.bias", 
"net.stage4_RB_2.net_bn_1.running_mean":"net.stage4.2.net.1.running_mean", 
"net.stage4_RB_2.net_bn_1.running_var":"net.stage4.2.net.1.running_var", 
"net.stage4_RB_2.net_conv3d_2.kernel":"net.stage4.2.net.3.kernel", 
"net.stage4_RB_2.net_bn_3.weight":"net.stage4.2.net.4.weight", 
"net.stage4_RB_2.net_bn_3.bias":"net.stage4.2.net.4.bias", 
"net.stage4_RB_2.net_bn_3.running_mean":"net.stage4.2.net.4.running_mean", 
"net.stage4_RB_2.net_bn_3.running_var":"net.stage4.2.net.4.running_var",

"net.up1_BC_0.net_conv3d_0.kernel":"net.up1.0.net.0.kernel",
"net.up1_BC_0.net_bn_1.weight":"net.up1.0.net.1.weight",
"net.up1_BC_0.net_bn_1.bias":"net.up1.0.net.1.bias",
"net.up1_BC_0.net_bn_1.running_mean":"net.up1.0.net.1.running_mean",
"net.up1_BC_0.net_bn_1.running_var":"net.up1.0.net.1.running_var",
"net.up1_2_RB_1.net_conv3d_0.kernel":"net.up1.1.0.net.0.kernel",
"net.up1_2_RB_1.net_bn_1.weight":"net.up1.1.0.net.1.weight",
"net.up1_2_RB_1.net_bn_1.bias":"net.up1.1.0.net.1.bias",
"net.up1_2_RB_1.net_bn_1.running_mean":"net.up1.1.0.net.1.running_mean",
"net.up1_2_RB_1.net_bn_1.running_var":"net.up1.1.0.net.1.running_var",
"net.up1_2_RB_1.net_conv3d_2.kernel":"net.up1.1.0.net.3.kernel",
"net.up1_2_RB_1.net_bn_3.weight":"net.up1.1.0.net.4.weight",

"net.up1_2_RB_1.net_bn_3.bias":	"net.up1.1.0.net.4.bias",
"net.up1_2_RB_1.net_bn_3.running_mean":	"net.up1.1.0.net.4.running_mean",
"net.up1_2_RB_1.net_bn_3.running_var":	"net.up1.1.0.net.4.running_var",
"net.up1_2_RB_1.downsample_conv3d_0.kernel":	"net.up1.1.0.downsample.0.kernel",
"net.up1_2_RB_1.downsample_bn_1.weight":	"net.up1.1.0.downsample.1.weight",
"net.up1_2_RB_1.downsample_bn_1.bias":	"net.up1.1.0.downsample.1.bias",
"net.up1_2_RB_1.downsample_bn_1.running_mean":	"net.up1.1.0.downsample.1.running_mean",
"net.up1_2_RB_1.downsample_bn_1.running_var":	"net.up1.1.0.downsample.1.running_var",
"net.up1_2_RB_2.net_conv3d_0.kernel":	"net.up1.1.1.net.0.kernel",
"net.up1_2_RB_2.net_bn_1.weight":	"net.up1.1.1.net.1.weight",
"net.up1_2_RB_2.net_bn_1.bias":	"net.up1.1.1.net.1.bias",
"net.up1_2_RB_2.net_bn_1.running_mean":	"net.up1.1.1.net.1.running_mean",
"net.up1_2_RB_2.net_bn_1.running_var":	"net.up1.1.1.net.1.running_var",
"net.up1_2_RB_2.net_conv3d_2.kernel":	"net.up1.1.1.net.3.kernel",
"net.up1_2_RB_2.net_bn_3.weight":	"net.up1.1.1.net.4.weight",
"net.up1_2_RB_2.net_bn_3.bias":	"net.up1.1.1.net.4.bias",
"net.up1_2_RB_2.net_bn_3.running_mean":	"net.up1.1.1.net.4.running_mean",
"net.up1_2_RB_2.net_bn_3.running_var":	"net.up1.1.1.net.4.running_var",

"net.up2_BC_0.net_conv3d_0.kernel":	"net.up2.0.net.0.kernel",
"net.up2_BC_0.net_bn_1.weight":	"net.up2.0.net.1.weight",
"net.up2_BC_0.net_bn_1.bias":	"net.up2.0.net.1.bias",
"net.up2_BC_0.net_bn_1.running_mean":	"net.up2.0.net.1.running_mean",
"net.up2_BC_0.net_bn_1.running_var":	"net.up2.0.net.1.running_var",
"net.up2_2_RB_1.net_conv3d_0.kernel":	"net.up2.1.0.net.0.kernel",
"net.up2_2_RB_1.net_bn_1.weight":	"net.up2.1.0.net.1.weight",
"net.up2_2_RB_1.net_bn_1.bias":	"net.up2.1.0.net.1.bias",
"net.up2_2_RB_1.net_bn_1.running_mean":	"net.up2.1.0.net.1.running_mean",
"net.up2_2_RB_1.net_bn_1.running_var":	"net.up2.1.0.net.1.running_var",
"net.up2_2_RB_1.net_conv3d_2.kernel":	"net.up2.1.0.net.3.kernel",
"net.up2_2_RB_1.net_bn_3.weight":	"net.up2.1.0.net.4.weight",
"net.up2_2_RB_1.net_bn_3.bias":	"net.up2.1.0.net.4.bias",
"net.up2_2_RB_1.net_bn_3.running_mean":	"net.up2.1.0.net.4.running_mean",
"net.up2_2_RB_1.net_bn_3.running_var":	"net.up2.1.0.net.4.running_var",
"net.up2_2_RB_1.downsample_conv3d_0.kernel":"net.up2.1.0.downsample.0.kernel",
"net.up2_2_RB_1.downsample_bn_1.weight":	"net.up2.1.0.downsample.1.weight",
"net.up2_2_RB_1.downsample_bn_1.bias":	"net.up2.1.0.downsample.1.bias",
"net.up2_2_RB_1.downsample_bn_1.running_mean":	"net.up2.1.0.downsample.1.running_mean",
"net.up2_2_RB_1.downsample_bn_1.running_var":	"net.up2.1.0.downsample.1.running_var",
"net.up2_2_RB_2.net_conv3d_0.kernel":	"net.up2.1.1.net.0.kernel",
"net.up2_2_RB_2.net_bn_1.weight":	"net.up2.1.1.net.1.weight",
"net.up2_2_RB_2.net_bn_1.bias":	"net.up2.1.1.net.1.bias",
"net.up2_2_RB_2.net_bn_1.running_mean":	"net.up2.1.1.net.1.running_mean",
"net.up2_2_RB_2.net_bn_1.running_var":	"net.up2.1.1.net.1.running_var",
"net.up2_2_RB_2.net_conv3d_2.kernel":	"net.up2.1.1.net.3.kernel",
"net.up2_2_RB_2.net_bn_3.weight":	"net.up2.1.1.net.4.weight",
"net.up2_2_RB_2.net_bn_3.bias":	"net.up2.1.1.net.4.bias",
"net.up2_2_RB_2.net_bn_3.running_mean":	"net.up2.1.1.net.4.running_mean",
"net.up2_2_RB_2.net_bn_3.running_var":	"net.up2.1.1.net.4.running_var",

"net.up3_BC_0.net_conv3d_0.kernel":	"net.up3.0.net.0.kernel",
"net.up3_BC_0.net_bn_1.weight":	"net.up3.0.net.1.weight",
"net.up3_BC_0.net_bn_1.bias":	"net.up3.0.net.1.bias",
"net.up3_BC_0.net_bn_1.running_mean":	"net.up3.0.net.1.running_mean",
"net.up3_BC_0.net_bn_1.running_var":	"net.up3.0.net.1.running_var",
"net.up3_2_RB_1.net_conv3d_0.kernel":	"net.up3.1.0.net.0.kernel",
"net.up3_2_RB_1.net_bn_1.weight":	"net.up3.1.0.net.1.weight",
"net.up3_2_RB_1.net_bn_1.bias":	"net.up3.1.0.net.1.bias",
"net.up3_2_RB_1.net_bn_1.running_mean":	"net.up3.1.0.net.1.running_mean",
"net.up3_2_RB_1.net_bn_1.running_var":	"net.up3.1.0.net.1.running_var",
"net.up3_2_RB_1.net_conv3d_2.kernel":	"net.up3.1.0.net.3.kernel",
"net.up3_2_RB_1.net_bn_3.weight":	"net.up3.1.0.net.4.weight",
"net.up3_2_RB_1.net_bn_3.bias":	"net.up3.1.0.net.4.bias",
"net.up3_2_RB_1.net_bn_3.running_mean":	"net.up3.1.0.net.4.running_mean",
"net.up3_2_RB_1.net_bn_3.running_var":	"net.up3.1.0.net.4.running_var",
"net.up3_2_RB_1.downsample_conv3d_0.kernel":	"net.up3.1.0.downsample.0.kernel",
"net.up3_2_RB_1.downsample_bn_1.weight":	"net.up3.1.0.downsample.1.weight",
"net.up3_2_RB_1.downsample_bn_1.bias":	"net.up3.1.0.downsample.1.bias",
"net.up3_2_RB_1.downsample_bn_1.running_mean":	"net.up3.1.0.downsample.1.running_mean",
"net.up3_2_RB_1.downsample_bn_1.running_var":	"net.up3.1.0.downsample.1.running_var",
"net.up3_2_RB_2.net_conv3d_0.kernel":	"net.up3.1.1.net.0.kernel",
"net.up3_2_RB_2.net_bn_1.weight":	"net.up3.1.1.net.1.weight",
"net.up3_2_RB_2.net_bn_1.bias":	"net.up3.1.1.net.1.bias",
"net.up3_2_RB_2.net_bn_1.running_mean":	"net.up3.1.1.net.1.running_mean",
"net.up3_2_RB_2.net_bn_1.running_var":	"net.up3.1.1.net.1.running_var",
"net.up3_2_RB_2.net_conv3d_2.kernel":	"net.up3.1.1.net.3.kernel",
"net.up3_2_RB_2.net_bn_3.weight":	"net.up3.1.1.net.4.weight",
"net.up3_2_RB_2.net_bn_3.bias":	"net.up3.1.1.net.4.bias",
"net.up3_2_RB_2.net_bn_3.running_mean":	"net.up3.1.1.net.4.running_mean",
"net.up3_2_RB_2.net_bn_3.running_var":	"net.up3.1.1.net.4.running_var",

"net.up4_BC_0.net_conv3d_0.kernel":	"net.up4.0.net.0.kernel",
"net.up4_BC_0.net_bn_1.weight":	"net.up4.0.net.1.weight",
"net.up4_BC_0.net_bn_1.bias":	"net.up4.0.net.1.bias",
"net.up4_BC_0.net_bn_1.running_mean":	"net.up4.0.net.1.running_mean",
"net.up4_BC_0.net_bn_1.running_var":	"net.up4.0.net.1.running_var",
"net.up4_2_RB_1.net_conv3d_0.kernel":	"net.up4.1.0.net.0.kernel",
"net.up4_2_RB_1.net_bn_1.weight":	"net.up4.1.0.net.1.weight",
"net.up4_2_RB_1.net_bn_1.bias":	"net.up4.1.0.net.1.bias",
"net.up4_2_RB_1.net_bn_1.running_mean":	"net.up4.1.0.net.1.running_mean",
"net.up4_2_RB_1.net_bn_1.running_var":	"net.up4.1.0.net.1.running_var",
"net.up4_2_RB_1.net_conv3d_2.kernel":	"net.up4.1.0.net.3.kernel",
"net.up4_2_RB_1.net_bn_3.weight":	"net.up4.1.0.net.4.weight",
"net.up4_2_RB_1.net_bn_3.bias":	"net.up4.1.0.net.4.bias",
"net.up4_2_RB_1.net_bn_3.running_mean":	"net.up4.1.0.net.4.running_mean",
"net.up4_2_RB_1.net_bn_3.running_var":	"net.up4.1.0.net.4.running_var",
"net.up4_2_RB_1.downsample_conv3d_0.kernel":"net.up4.1.0.downsample.0.kernel",
"net.up4_2_RB_1.downsample_bn_1.weight":	"net.up4.1.0.downsample.1.weight",
"net.up4_2_RB_1.downsample_bn_1.bias":	"net.up4.1.0.downsample.1.bias",
"net.up4_2_RB_1.downsample_bn_1.running_mean":	"net.up4.1.0.downsample.1.running_mean",
"net.up4_2_RB_1.downsample_bn_1.running_var":	"net.up4.1.0.downsample.1.running_var",
"net.up4_2_RB_2.net_conv3d_0.kernel":	"net.up4.1.1.net.0.kernel",
"net.up4_2_RB_2.net_bn_1.weight":	"net.up4.1.1.net.1.weight",
"net.up4_2_RB_2.net_bn_1.bias":	"net.up4.1.1.net.1.bias",
"net.up4_2_RB_2.net_bn_1.running_mean":	"net.up4.1.1.net.1.running_mean",
"net.up4_2_RB_2.net_bn_1.running_var":	"net.up4.1.1.net.1.running_var",
"net.up4_2_RB_2.net_conv3d_2.kernel":	"net.up4.1.1.net.3.kernel",
"net.up4_2_RB_2.net_bn_3.weight":	"net.up4.1.1.net.4.weight",
"net.up4_2_RB_2.net_bn_3.bias":	"net.up4.1.1.net.4.bias",
"net.up4_2_RB_2.net_bn_3.running_mean":	"net.up4.1.1.net.4.running_mean",
"net.up4_2_RB_2.net_bn_3.running_var":	"net.up4.1.1.net.4.running_var"}

config={
    "fast_rep_flag": False,
    # Dataset configs
    "dataset_name" : 'UnknwonDataset',
    "dataset_root" : 'UnknownDatasetPath',
    "source_dataset_name" : 'UnknwonDataset',
    "source_dataset_root" : 'UnknownDatasetPath',
    "target_dataset_name" : 'UnknwonDataset',
    "target_dataset_root" : 'UnknownDatasetPath',
    "save_dir" : 'results_ckpt',
    "ns_dataset_version":'v1.0-trainval',
    #"ns_dataset_version":'v1.0-mini',

    # splits
    "train_split" : 'train',
    "val_split" : 'val',
    "test_split" : 'val',
    "nb_classes" : 1,

    # Method parameters
    "input_intensities" : False,
    "input_dirs":False,
    "input_normals":False,
    "source_input_intensities":False,
    "source_input_dirs": False,    
    "target_input_intensities":False,
    "target_input_dirs" :False,

    "manifold_points" : 10000,
    "non_manifold_points" :2048,
    
    # Training parameters
    "da_flag": False,
    "dual_seg_head" : True,
    "training_iter_nbr" : 50000,
    "training_batch_size" : 4,
    "test_batch_size" : 1,
    "training_lr_start" : 0.001,
    "training_lr_start_head" : None,
    "optimizer" : "AdamW",
    "lr_scheduler" : None,
    "step_lr_step_size" :200000, #Step size for steplr scheduler
    "step_lr_gamma" : 0.7, #Gamma for steplr scheduler
    "voxel_size" : 0.1,
    "val_interval" :5,
    "resume" :False,

    # Network parameter
    "network_backbone" : 'TorchSparseMinkUNet_inside',
    "network_latent_size": 128,
    "network_decoder" : 'InterpAllRadiusNet',
    "network_decoder_k" : 2.0,
    "network_n_labels" : 1,
    "use_no_dirs_rec_head_flag" : False,
    "rotation_head_flag" : False,

    # Technical parameter
    "device" : 'cuda',
    "threads" :6,
    "interactive_log" : True,
    "logging" : 'INFO',
    "use_amp" : False,

    # Data augmentation
    "randRotationZ" : True,
    "randFlip" : True,
    "no_augmentation" : False,
    
    # Ckpt path 
    "ckpt_path_model" : "UnknownPath",

 
    # Ignorance idx
    "ignore_idx" : 0,
    "get_latent" : False,


    # Test flag
    "test_flag_eval" : False,
    "target_training" : True,
    "source_training" : True,

    # Which ckpt to load from in eval
    "ckpt_number" : -1,

    
    "da_flag" : True,
    "dual_seg_head" : True,
    "source_dataset_name" : 'NuScenes',
    "source_dataset_root" : 'data/nuscenes',
    "source_input_intensities" : False,
    "source_input_dirs" : False,
    "nb_classes" : 11,
    "target_dataset_name" : 'SemanticKITTI',
    "target_dataset_root" : 'data/SemanticKITTI',
    "target_input_intensities" : False,
    "target_input_dirs" : False,
    "lr_scheduler":"step_lr",
    "training_iter_nbr":200000,
    "step_lr_gamma":0.7,
    
}


def read_yaml_file(file_path) -> dict:
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error while reading the YAML file: {e}")
            return {}