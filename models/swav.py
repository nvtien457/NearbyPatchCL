import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes=0):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out

class SwAV(nn.Module):
    def __init__(self, backbone, hidden_dim=2048, proj_dim=128, nmb_prototypes=3000, normalize=False):
        super().__init__()
        self.backbone = backbone

        # normalize output features
        self.l2norm = normalize

        # projector layer
        if proj_dim == 0:
            self.projection_head = None
        elif hidden_dim == 0:
            self.projection_head = nn.Linear(self.backbone.output_dim, proj_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(self.backbone.output_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, proj_dim),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(proj_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(proj_dim, nmb_prototypes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_backbone(self, x):
        return self.backbone(x)

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        # print(len(inputs))
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        # print(torch.unique_consecutive(
        #     torch.tensor([inp.shape[-1] for inp in inputs]),
        #     return_counts=True,
        # ))
        # print(idx_crops)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)