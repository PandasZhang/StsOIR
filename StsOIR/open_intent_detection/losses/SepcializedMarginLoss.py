import torch
import math
import torch.nn.functional as F
from torch import nn

class SepcializedMarginLoss(nn.Module):

    """
    cos_theta and target need to be normalized first
    """ 

    def __init__(self, m=0.35, s=30):
        
        super(SepcializedMarginLoss, self).__init__()
        self.m = m
        self.s = s

    '''
    args:
        logits_prob: results of model predict. shape = [batch, num_labels]
        target: true label of samples. shape = [batch]
    
    '''
    def forward(self, logits_prob, target):
        deff = logits_prob[ F.one_hot(target) == 1 ][:, None]-logits_prob[ F.one_hot(target) != 1 ].reshape(len(target),-1)
        # loss = -deff.log().mean()
        return -deff.log().mean()