import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LALoss(nn.Module):
    ## logit adjustement loss function

    def __init__(self, cls_num_list, tau=1.0):
        super(LALoss, self).__init__()
        base_probs = cls_num_list / cls_num_list.sum()
        scaled_class_weights = tau * torch.log(base_probs + 1e-12)
        scaled_class_weights = scaled_class_weights.reshape(1, -1)  # [1,classnum]
        self.scaled_class_weights = scaled_class_weights.float().cuda()

    def forward(self, x, target):
        x += self.scaled_class_weights
        return F.cross_entropy(x, target)
