import torch.nn as nn
import torch.nn.functional as F

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, backbone, head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        self.backbone = backbone
        in_dim = backbone.output_dim

        if head == 'linear':
            self.head = nn.Linear(in_dim, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.backbone(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat