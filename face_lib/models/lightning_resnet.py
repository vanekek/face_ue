import torch


import sys
sys.path.append("/app")
from face_lib import models as mlib


class ResNet(torch.nn.Module):
    def __init__(self, resnet_name: str, weights: str, learnable: bool) -> None:
        self.backbone = mlib.model_dict[resnet_name](learnable=learnable)
        
        if weights is not None:
            backbone_dict = torch.load(
                self.model_args.pretrained_backbone, map_location=self.device
            )
            self.backbone.load_state_dict(backbone_dict)
    def forward(self, x):
        return self.backbone(x)