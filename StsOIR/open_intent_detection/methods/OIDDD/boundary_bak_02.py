import torch
from torch import nn
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    """
    Deep Open Intent Classification with Adaptive Decision Boundary.
    https://arxiv.org/pdf/2012.10209.pdf
    """
    def __init__(self, num_labels=10, feat_dim=2):
        super(BoundaryLoss, self).__init__()

        self.param_dim = 2

        self.num_labels = num_labels
        self.feat_dim = feat_dim
        self.delta = nn.Parameter(torch.randn(num_labels).cuda())
        nn.init.normal_(self.delta, mean=2)
        '''
            单向改多向
        '''
        # self.param_a = nn.Parameter(torch.randn(num_labels).cuda())
        # self.param_b = nn.Parameter(torch.randn(num_labels).cuda())
        self.param_ab = nn.Parameter(torch.randn(num_labels, self.param_dim ).cuda())
        nn.init.normal_(self.param_ab, mean=1)
        # nn.init.normal_(self.param_a, mean=1)
        # nn.init.normal_(self.param_b, mean=1)
        
    def forward(self, pooled_output, centroids, labels, w = 1):
        
        # pA = F.softplus(self.param_a[labels])
        # pB = F.softplus(self.param_b[labels])
        delta = F.softplus(self.delta)
        # pA = self.param_a[labels]
        # pB = self.param_b[labels]
        # pA = pA*pA
        # pB = pB*pB
        k = F.softplus(self.param_ab[labels])
        d = delta[labels]
        c = centroids[labels]
        x = pooled_output
        '''
            单向改多向
        '''
        z = (x-c).view(x.shape[0], self.param_dim, x.shape[1]//self.param_dim)
        z = torch.norm(z, 2, 2)
        euc_dis = z.mul(k).sum(-1)
        # euc_dis = z.mul(z).mul(pAB).sum(-1)
        # euc_dis = torch.norm(x - c,2, 1).view(-1)
        pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
        neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)
        
        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask 
        # loss = w * pos_loss.mean()
        loss = w * pos_loss.mean() + neg_loss.mean()
        
        return loss, delta 

