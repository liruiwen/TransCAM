# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

from deeplab.utils.registry import BACKBONES

def build_backbone(backbone_name, pretrained=True, **kwargs):
	net = BACKBONES.get(backbone_name)(pretrained=pretrained, **kwargs)
	return net
