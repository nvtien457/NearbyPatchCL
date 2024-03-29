'''
Modify from https://github.com/Spijkervet/SimCLR

Because we train with only 1 GPU,
distributed training with multi-gpu is not tested

Assume: world_size = 1 
'''
import torch
import torch.nn as nn
import torch.distributed as dist

__all__ = ["GatherLayer", "NT_Xent"]

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
    

class NT_Xent(nn.Module):
    def __init__(self, temperature, world_size=1):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.world_size = world_size

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size, device=torch.device("cuda")):
        '''
        Generate mask for negative samples.
        With batch_size = B samples after 2 augmentations (aug), there are 2B aug-samples.

            For example:    aug-batch = ( x_1, x_2, ..., x_B, x_1', x_2', ..., x_B' )
                                          \----------------/  \------------------/
                                              first aug             second-aug

                            mask = [
                                0, 1, .., 1,   0, 1, .., 1,             # 
                                1, 0, .., 1,   1, 0, .., 0,             #
                                .                                       #   first-aug, B rows
                                .                                       #
                                1, 1, .., 0,   1, 1, .., 0,             #

                                0, 1, .., 1,   0, 1, .., 1,             # 
                                1, 0, .., 1,   1, 0, .., 0,             #
                                .                                       #   second-aug, B rows
                                .                                       #
                                1, 1, .., 0,   1, 1, .., 0,             #
                            ]
                                \---------/    \---------/
                                 first-aug      second-aug
                                  B cols         B cols

                        
        For each aug-sample, 
            Main diagonal is removed
            2B     positives
            2B - 2 negatives
        '''
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool, device=device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):    # remove other aug from negative samples (optional)
            mask[i, batch_size * world_size + i] = 0    
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]               # B
        N = 2 * batch_size * self.world_size    # N
        device = z_i.device

        mask = self.mask_correlated_samples(batch_size, self.world_size, device)    # only negative sample (different samples in batch)
        print(mask)

        if self.world_size > 1:
            z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
        z = torch.cat((z_i, z_j), dim=0)        # (2B x D)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature  # cosine similarity matrix (2B x 2B)

        # positive samples
        sim_i_j = torch.diag(sim, batch_size * self.world_size)     # Similarity between aug1 x aug2 (B)
        sim_j_i = torch.diag(sim, -batch_size * self.world_size)    # Similarity between aug2 x aug1 (B)

        # We have 2B samples, but with Distributed training every GPU gets B examples too, resulting in: 2xBxB
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)   # (2B x 1)
        negative_samples = sim[mask].reshape(N, -1)                             # (2B x (2B-2))

        labels = torch.zeros(N, device=device, dtype=torch.long)            # 2B x 2B
        logits = torch.cat((positive_samples, negative_samples), dim=1)     # 2B x 2B-1
        loss = self.criterion(logits, labels)
        loss /= N

        return loss
