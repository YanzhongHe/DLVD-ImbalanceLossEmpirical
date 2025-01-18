import torch
import torch.nn.functional as F
import math

def PHuberCrossEntropy(input, target, tau: float = 10, reduction: str = 'mean'):
    prob_thresh = 1 / tau
    boundary_term = math.log(tau) + 1

    p = F.softmax(input, dim=-1)
    p = p[torch.arange(p.shape[0]), target]

    loss = torch.empty_like(p)
    clip = p <= prob_thresh

    loss[clip] = -tau * p[clip] + boundary_term
    loss[~clip] = -torch.log(p[~clip])
    if reduction == 'none':
        return loss

    return torch.mean(loss)