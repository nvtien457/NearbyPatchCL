import torch
import torch.nn as nn

class MICLe(nn.Module):
    def __init__(self, backbone, dim=128):
        super(MICLe, self).__init__()
 
        self.backbone = backbone

        self.feature_dim = backbone.output_dim
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim, bias=False),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, dim, bias=True),
        )

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z