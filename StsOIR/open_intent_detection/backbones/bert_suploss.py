import torch
import math
import torch.nn.functional as F
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch.nn.parameter import Parameter
from .utils import L2_normalization
# from ..losses.SupConLoss import SupConLoss
activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'elu': nn.ELU()}

class BERT_SUP(BertPreTrainedModel):

    def __init__(self, config, args):

        super(BERT_SUP, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        # self.loss_fct = 
        self.activatation_softsign = nn.Softsign()
        '''
        TODO
        '''
        # self.test_loss = nn.SoftMarginLoss
        self.k = 20
        # self.k = args.tmp_k
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None, centroids=None, deltas=None, use_dis=False):

        encoded_layer_12, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.activatation_softsign(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                # return self.loss_fct(logits, labels)
                loss = loss_fct(logits, labels)
                if centroids is not None:
                    # loss_dis = dis_loss(pooled_output, centroids, labels)
                    loss_dis = dis_koss_2_bak(pooled_output, centroids, labels, deltas)
                    # loss = self.loss_fct(logits, labels)
                    return loss + self.k * loss_dis
                    return self.k * loss + (1-self.k)* loss_dis
                return loss
            else:
                return pooled_output, logits

def dis_loss(feats, centroids, labels):
    c = centroids[labels]
    dis = torch.norm(feats - c , 2, 1).view(-1)
    # loss = F.sigmoid(dis)
    loss = dis.mean()
    return loss

def dis_koss_2(feats, centroids, labels, deltas):
    c = centroids[labels]
    d = deltas[labels]
    dis = torch.norm( feats - c, 2, 1 ).view(-1)
    pos_mask = (dis > d).type(torch.cuda.FloatTensor)
    neg_mask = (dis < d).type(torch.cuda.FloatTensor)
    pos_loss = (dis - d) * pos_mask
    neg_loss = (d - dis) * neg_mask
    loss =  pos_loss.mean() + neg_loss.mean()
    return loss

def dis_koss_2_bak(feats, centroids, labels, deltas):
    c = centroids[labels]
    d = deltas[labels]
    dis = torch.norm( feats - c, 2, 1 ).view(-1)
    pos_mask = (dis > d).type(torch.cuda.FloatTensor)
    pos_loss = (dis - d) * pos_mask
    return pos_loss.mean()

class BERT_SUP_Norm(BertPreTrainedModel):

    def __init__(self, config, args):

        super(BERT_SUP_Norm, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        from torch.nn.utils import weight_norm
        self.norm = L2_normalization()
        self.classifier = weight_norm(
            nn.Linear(config.hidden_size, args.num_labels), name='weight')
        self.k = 20
        # if args.tmp_k is not None:
        #     self.k = args.tmp_k
        self.activatation_relu = nn.ReLU()
        self.activatation_softsign = nn.Softsign()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None, centroids=None, deltas=None, use_dis=False):

        encoded_layer_12, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
        '''添加两个激活层'''
        pooled_output = self.activatation_relu(pooled_output)
        pooled_output = self.activatation_softsign(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.norm(pooled_output)
        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss = loss_fct(logits, labels)
                if centroids is not None and use_dis:
                    # loss_dis = dis_loss(pooled_output, centroids, labels)
                    loss_dis = dis_koss_2_bak(pooled_output, centroids, labels, deltas)
                    # loss = self.loss_fct(logits, labels)
                    # return self.k * loss + (1-self.k)* loss_dis
                    return  loss + self.k * loss_dis
                return loss
            else:
                return pooled_output, logits
