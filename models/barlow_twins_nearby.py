import torch
import torch.nn as nn
import torch.nn.functional as F

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins_nearby(nn.Module):
    def __init__(self, backbone, projector='8192-8192-8192', lambd=0.0051):
        super().__init__()
        self.backbone = backbone
        self.lambd = lambd

        # projector
        sizes = [2048] + list(map(int, projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2, yn):
        batch_size = y1.shape[0]

        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))
        zn = self.projector(self.backbone(yn))
        dim = z1.shape[1]
        D= z2.size()[-1]
        ##################################
        # size=128
        # z1_bn = self.bn(z1)
        # z2_bn = self.bn(z2)
        # zn_bn = self.bn(zn)

        # D= z2_bn.size()[-1]
        # total_loss =0
        
        # for i in range (D//size ):
        #     for j in range( D//size):
        #         c1= z1_bn[  : ,i*size: (i+1)*size ].T @ z2_bn[ :, j*size: (j+1)*size]
        #         c1.div_(batch_size)
        #         c2= z1_bn[  : ,i*size: (i+1)*size ].T @ zn_bn[ :, j*size: (j+1)*size]
        #         c2.div_(batch_size)
        #         if i == j:
        #             on_diag1 = torch.diagonal(c1).add_(-1).pow_(2).sum()
        #             on_diag2 = torch.diagonal(c2).add_(-1).pow_(2).sum()
        #         else:
        #             on_diag1 = self.lambd *torch.diagonal(c1).pow_(2).sum()
        #             on_diag2 = self.lambd *torch.diagonal(c2).pow_(2).sum()
        #         off_diag1 = off_diagonal(c1).pow_(2).sum()
        #         off_diag2 = off_diagonal(c2).pow_(2).sum()
        #         loss = 0.8*(on_diag1 + self.lambd * off_diag1) + 0.2*(on_diag2 + self.lambd * off_diag2)
        #         total_loss += loss
        #         del c
        
        ######################
        
        # empirical cross-correlation matrix
        c1 = self.bn(z1).T @ self.bn(z2)
        c2 = self.bn(z1).T @ self.bn(zn)
        # sum the cross-correlation matrix between all gpus
        c1.div_(batch_size)
        c2.div_(batch_size)
        # torch.distributed.all_reduce(c)

        on_diag1 = torch.diagonal(c1).add_(-1).pow_(2).sum()
        off_diag1 = off_diagonal(c1).pow_(2).sum()

        on_diag2 = torch.diagonal(c2).add_(-1).pow_(2).sum()
        off_diag2 = off_diagonal(c2).pow_(2).sum()
        loss = 0.8*(on_diag1 + self.lambd * off_diag1 ) + 0.2*(on_diag2 +self.lambd * off_diag2)

        return z1, z2, loss*128/D