from tools import accuracy
import torch
import torch.nn.functional as F

B = 8
D = 4

a = torch.randn((B, D))

logit = torch.mm(F.normalize(a, dim=-1), F.normalize(a, dim=-1).T)
target = torch.arange(0, B, dtype=torch.long)

acc1 = accuracy(logit, target, topk=(1,))

print(acc1)