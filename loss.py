import torch
import torch.nn.functional as F

class NT_Xent_Loss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NT_Xent_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # Compute similarity matrix
        sim = torch.matmul(z_i, z_j.T) / self.temperature
        labels = torch.arange(z_i.size(0)).cuda()
        loss = F.cross_entropy(sim, labels)
        return loss
