import torch
import time
from losses import SupConLoss, NT_Xent, SCLoss

B = 2
D = 3

criterion = SCLoss(temperature=0.07)
ntx = NT_Xent(temperature=0.07)

a1 = torch.randn((B, D))
a2 = torch.randn((B, D))
an = torch.randn((B, D))

loss = criterion(a1, a2, an)
print(loss)

loss = ntx(a1, a2) + ntx(a1, an) + ntx(a2, an)
print(loss)

# logits = torch.mm(torch.cat((a1, a2), dim=0), torch.cat((a1, a2, an), dim=0).T)

# print(logits.shape)
# print(logits)

# print(torch.diag(logits, diagonal=0))
# print(torch.diag(logits, diagonal=B))
# print(torch.diag(logits, diagonal=2*B))
# print(torch.diag(logits, diagonal=-B))