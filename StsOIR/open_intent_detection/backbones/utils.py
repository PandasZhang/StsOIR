import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class L2_normalization(nn.Module):
    def forward(self, input):
        return l2_norm(input)   

def freeze_bert_parameters(model):
    
    for name, param in model.bert.named_parameters():  
        param.requires_grad = False
        if "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True
    return model


def to_one_hot(inp,num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    return Variable(y_onehot.cuda(),requires_grad=False)

def mixup_x(out, lam):
    indices = np.random.permutation(out.size(0))
    indices_2 = np.random.permutation(out.size(0))
    # out = out*lam + out[indices]*(1-lam) 
    out = out*lam + out[indices]*(1-lam) 
    # out = out*lam + out[indices]*(1-lam) * 0.5 + out[indices_2]*(1-lam) * 0.5 
    # out = out*lam + out[indices]*(1-lam) * 0.5 + out[indices_2]*(1-lam) * 0.2 + out[indices_3]*(1-lam) * 0.2 + out[indices_4]*(1-lam) * 0.1
    return out, indices


def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    indices_2 = np.random.permutation(out.size(0))
    # out = out*lam + out[indices]*(1-lam) 
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    out = out*lam + out[indices]*(1-lam) * 0.5 + out[indices_2]*(1-lam) * 0.5 
    # out = out*lam + out[indices]*(1-lam) * 0.5 + out[indices_2]*(1-lam) * 0.2 + out[indices_3]*(1-lam) * 0.2 + out[indices_4]*(1-lam) * 0.1
    # return out, indices
    return out, target_reweighted, indices
def mixup_process_0621(out, target_reweighted, lam):
    import copy
    indices = np.random.permutation(out.size(0))
    out_bak = copy.deepcopy(out)
    out_bak = out*lam + out_bak[indices]*(1-lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    
    return out, target_reweighted, indices