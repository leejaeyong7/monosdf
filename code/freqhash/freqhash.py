import torch
import torch.nn as nn

class FreqHash(nn.Module):
    def __init__(self, log2_res=5, num_encodings=6, num_feats=16, std=0.2):
        res = 2 ** log2_res
        features = torch.randn((num_encodings * 2 * 3, num_feats, res, 1)) * std
        self.features = nn.Parameter(features, True)
        return

    def forward(enc_pos):

        return