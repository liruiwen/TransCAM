import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

np.set_printoptions(threshold=np.inf)

import network.conformer


class Net_ti(network.conformer.Net):
    def __init__(self):
        super(Net_ti, self).__init__(patch_size=16, channel_ratio=1, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, drop_rate=0.0, drop_path_rate=0.1, num_classes=21)

class Net_sm(network.conformer.Net):
    def __init__(self):
        super(Net_sm, self).__init__(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, drop_rate=0.0, drop_path_rate=0.1, num_classes=21)

class Net_bs(network.conformer.Net):
    def __init__(self):
        super(Net_bs, self).__init__(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                      num_heads=9, mlp_ratio=4, qkv_bias=True, drop_rate=0.0, drop_path_rate=0.1, num_classes=21)
