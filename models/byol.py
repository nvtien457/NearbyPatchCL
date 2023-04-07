""" BYOL Model """

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

import warnings
import copy

import torch
import torch.nn as nn

class BYOLProjectionHead(nn.Module):
    """Projection head used for BYOL.
    "This MLP consists in a linear layer with output size 4096 followed by
    batch normalization, rectified linear units (ReLU), and a final
    linear layer with output dimension 256." [0]
    [0]: BYOL, 2020, https://arxiv.org/abs/2006.07733
    """

    def __init__(self, input_dim:int = 2048, hidden_dim:int = 4096, output_dim:int = 256):
        super(BYOLProjectionHead, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)
        )

    def forward(self, x:torch.Tensor):
        return self.layers(x)

def _deactivate_requires_grad(params):
    """Deactivates the requires_grad flag for all parameters."""
    for param in params:
        param.requires_grad = False


def _do_momentum_update(prev_params, params, m):
    """Updates the weights of the previous parameters."""
    for prev_param, param in zip(prev_params, params):
        prev_param.data = prev_param.data * m + param.data * (1.0 - m)


class BYOL(nn.Module):
    """Implementation of the BYOL architecture.
    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection mlp).
        hidden_dim:
            Dimension of the hidden layer in the projection and prediction mlp.
        out_dim:
            Dimension of the output (after the projection/prediction mlp).
        m:
            Momentum for the momentum update of encoder.
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_dim: int = 2048,
        hidden_dim: int = 4096,
        out_dim: int = 256,
        m: float = 0.9,
    ):
        super(BYOL, self).__init__()

        self.backbone = backbone
        # the architecture of the projection and prediction head is the same
        self.projection_head = BYOLProjectionHead(in_dim, hidden_dim, out_dim)
        self.prediction_head = BYOLProjectionHead(out_dim, hidden_dim, out_dim)
        self.momentum_backbone = None
        self.momentum_projection_head = None

        self._init_momentum_encoder()
        self.m = m

    def _init_momentum_encoder(self):
        """Initializes momentum backbone and a momentum projection head."""
        assert self.backbone is not None
        assert self.projection_head is not None

        self.momentum_backbone = copy.deepcopy(self.backbone)
        self.momentum_projection_head = copy.deepcopy(self.projection_head)

        _deactivate_requires_grad(self.momentum_backbone.parameters())
        _deactivate_requires_grad(self.momentum_projection_head.parameters())

    @torch.no_grad()
    def _momentum_update(self, m: float = 0.999):
        """Performs the momentum update for the backbone and projection head."""
        _do_momentum_update(
            self.momentum_backbone.parameters(),
            self.backbone.parameters(),
            m=m,
        )
        _do_momentum_update(
            self.momentum_projection_head.parameters(),
            self.projection_head.parameters(),
            m=m,
        )

    @torch.no_grad()
    def _batch_shuffle(self, batch: torch.Tensor):
        """Returns the shuffled batch and the indices to undo."""
        batch_size = batch.shape[0]
        shuffle = torch.randperm(batch_size, device=batch.device)
        return batch[shuffle], shuffle

    @torch.no_grad()
    def _batch_unshuffle(self, batch: torch.Tensor, shuffle: torch.Tensor):
        """Returns the unshuffled batch."""
        unshuffle = torch.argsort(shuffle)
        return batch[unshuffle]

    def _forward(self, x0: torch.Tensor, x1: torch.Tensor = None):
        """Forward pass through the encoder and the momentum encoder.
        Performs the momentum update, extracts features with the backbone and
        applies the projection (and prediciton) head to the output space. If
        x1 is None, only x0 will be processed otherwise, x0 is processed with
        the encoder and x1 with the momentum encoder.
        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
        Returns:
            The output proejction of x0 and (if x1 is not None) the output
            projection of x1.
        Examples:
            >>> # single input, single output
            >>> out = model._forward(x)
            >>>
            >>> # two inputs, two outputs
            >>> out0, out1 = model._forward(x0, x1)
        """

        self._momentum_update(self.m)

        # forward pass of first input x0
        f0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_head(f0)
        p0 = self.prediction_head(z0)

        if x1 is None:
            return p0

        # forward pass of second input x1
        with torch.no_grad():
            f1 = self.momentum_backbone(x1).flatten(start_dim=1)
            z1 = self.momentum_projection_head(f1)

        return p0, z1

    def forward(self, x0:torch.Tensor, x1:torch.Tensor):
        """Symmetrizes the forward pass (see _forward).
        Performs two forward passes, once where x0 is passed through the encoder
        and x1 through the momentum encoder and once the other way around.
        Note that this model currently requires two inputs for the forward pass
        (x0 and x1) which correspond to the two augmentations.
        Furthermore, `the return_features` argument does not work yet.
        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
        Returns:
            A tuple out0, out1, where out0 and out1 are tuples containing the
            predictions and projections of x0 and x1: out0 = (z0, p0) and
            out1 = (z1, p1).
        Examples:
            >>> # initialize the model and the loss function
            >>> model = BYOL()
            >>> criterion = SymNegCosineSimilarityLoss()
            >>>
            >>> # forward pass for two batches of transformed images x1 and x2
            >>> out0, out1 = model(x0, x1)
            >>> loss = criterion(out0, out1)
        """

        if x0 is None:
            raise ValueError("x0 must not be None!")
        if x1 is None:
            raise ValueError("x1 must not be None!")

        if not all([s0 == s1 for s0, s1 in zip(x0.shape, x1.shape)]):
            raise ValueError(
                f"x0 and x1 must have same shape but got shapes {x0.shape} and {x1.shape}!"
            )

        p0, z1 = self._forward(x0, x1)
        p1, z0 = self._forward(x1, x0)

        return (z0, p0), (z1, p1)