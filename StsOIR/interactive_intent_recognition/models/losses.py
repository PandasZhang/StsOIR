import torch
import torch.nn.functional as F

def contrastive_loss(z, labels, temperature=0.1):
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    sim_exp = torch.exp(sim) * (1 - torch.eye(z.size(0), device=z.device))
    pos_sim = torch.exp(sim) * mask
    loss = -torch.log(pos_sim.sum(1) / sim_exp.sum(1))
    return loss.mean()

def compute_prototypes(z, labels):
    unique = labels.unique()
    return torch.stack([z[labels == c].mean(dim=0) for c in unique])

def prototype_kl(P, Q):
    return F.kl_div(F.log_softmax(P, dim=1), F.softmax(Q, dim=1), reduction="batchmean")
