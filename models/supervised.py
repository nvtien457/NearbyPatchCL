import torch.nn as nn
import torch.nn.functional as F

class Supervised(nn.Module):
    """backbone + projection head"""
    def __init__(self, backbone, num_classes):
        super(Supervised, self).__init__()
        self.backbone = backbone
        self.linear = nn.Linear(in_features=backbone.output_dim, out_features=num_classes, bias=True)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.linear(feat)
        return feat