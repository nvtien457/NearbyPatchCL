# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, backbone, proj_dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        # self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # # build a 3-layer projector
        # proj_dim = self.encoder.fc.weight.shape[1]
        # self.encoder.fc = nn.Sequential(nn.Linear(proj_dim, proj_dim, bias=False),
        #                                 nn.BatchNorm1d(proj_dim),
        #                                 nn.ReLU(inplace=True), # first layer
        #                                 nn.Linear(proj_dim, proj_dim, bias=False),
        #                                 nn.BatchNorm1d(proj_dim),
        #                                 nn.ReLU(inplace=True), # second layer
        #                                 self.encoder.fc,
        #                                 nn.BatchNorm1d(dim, affine=False)) # output layer
        # self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN
        self.backbone = backbone
        in_dim = backbone.fc.in_features
        
        self.projector = nn.Sequential(
            nn.Linear(in_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),                      # first layer
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),                      # second layer
            nn.Linear(proj_dim, in_dim),
            nn.BatchNorm1d(in_dim, affine=False)           # third layer
        )
        
        self.projector[6].bias.requires_grad = False    # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(in_dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, in_dim)) # output layer

    def forward(self, x):
        """
        Input:
            x: a view of images
        Output:
            z: targets of the network (feature)
            p: predictors of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z = self.projector(self.backbone(x)) # NxC
        p = self.predictor(z) # NxC

        return z, p