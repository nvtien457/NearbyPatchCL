import torch.nn as nn
import torchvision

# https://github.com/Spijkervet/SimCLR/
class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, dim=128):
        super(SimCLR, self).__init__()

        self.backbone = encoder
        self.feature_dim = encoder.output_dim

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim, bias=False),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, dim, bias=True),
        )

    def forward(self, x):
        """
        Input:
            x: a view of images
        Output:
            z: feature 
            p: projection of feature
        """
        z = self.backbone(x)
        p = self.projector(z)

        return z, p