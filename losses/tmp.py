from NT_Xent import NT_Xent
import torch

B = 8
D = 56

a = torch.normal(0, 1, size=(B, D))
b = torch.normal(0, 1, size=(B, D))

criterion = NT_Xent(temperature=0.07)

loss = criterion(a, b)
print(loss)