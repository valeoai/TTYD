
def get_backbone(name):
    if name == "TorchSparseMinkUNet":
        from .torchsparse_minkunet import TorchSparseMinkUNet
        return TorchSparseMinkUNet
    elif name == "TorchSparseMinkUNet_learned":
        from .torchsparse_minkunet_learned import TorchSparseMinkUNet_learned
        return TorchSparseMinkUNet_learned
    else:
        raise ValueError("Unknown backbone")
