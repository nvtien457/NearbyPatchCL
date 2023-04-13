from models import MoCo, get_backbone
import torch

model = MoCo(backbone=get_backbone('resnet50')).cuda()

x = torch.randn((4, 16, 3, 224, 224))
y = torch.randn((4, 16, 3, 224, 224))

for i in range(4):
    xi = x[i].cuda()
    yi = y[i].cuda()
    logits, labels = model(xi, yi)