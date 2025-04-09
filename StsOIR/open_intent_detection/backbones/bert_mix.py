from cProfile import label
from fileinput import close
import torch
import math
import torch.nn.functional as F
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch.nn.parameter import Parameter
# from .utils import L2_normalization, get_lambda, mixup_data, mixup_process, to_one_hot
from .utils import L2_normalization,  mixup_x, to_one_hot, mixup_process

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'softplus': nn.Softplus(), 'softsign': nn.Softsign()}

class BertMix(BertPreTrainedModel):

    def __init__(self, config, args):

        super(BertMix, self).__init__(config)
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
        self.lam = 0.7
        self.loss_k = 0.5
        # self.k = args.tmp_k
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None, centroids=None, deltas=None, use_dis=False):

        if centroids is None and labels is not None:
            # pretrain采用mix混合的方式
            return self.mixForward(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels,
                feature_ext=feature_ext, mode=mode, loss_fct=loss_fct)
        # pretrain结束，第二阶段采用dis_loss强制收缩
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
    def mixForward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None, centroids=None, deltas=None, use_dis=False):
        
        encoded_layer_12, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        
        last_hidden_state = encoded_layer_12[-1]

        cated_hidden, indices = mixup_x(last_hidden_state, lam=self.lam)

        # labels_one_hot = None
        # if labels is not None:
        #     labels_one_hot = to_one_hot(labels, self.num_labels)
        # hidden_mixed, labels_mixed, indices = mixup_process(last_hidden_state, labels_one_hot, lam=self.lam)
        
        
        # cated_hidden = torch.cat((last_hidden_state, hidden_mixed), 0)
        # cated_labels = torch.cat((labels_one_hot, labels_mixed), 0)
        # cated_hidden = hidden_mixed
        # cated_labels = labels_mixed

        # pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
        pooled_output = self.dense(cated_hidden.mean(dim=1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.activatation_softsign(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        
        pooled_output = self.dense(last_hidden_state.mean(dim=1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.activatation_softsign(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits_ori = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                # return self.loss_fct(logits, labels)
                labels_one_hot = to_one_hot(labels, self.num_labels)
                cated_labels = self.lam * labels_one_hot + (1-self.lam)* labels_one_hot[indices]
                # cated_labels = torch.cat((labels_one_hot, labels_mixed), 0)
                loss = loss_fct(logits, cated_labels)
                loss_ori = loss_fct(logits_ori, labels)
                return self.loss_k * loss + (1-self.loss_k)* loss_ori
                # if centroids is not None:
                #     # loss_dis = dis_loss(pooled_output, centroids, labels)
                #     loss_dis = dis_koss_2_bak(pooled_output, centroids, labels, deltas)
                #     # loss = self.loss_fct(logits, labels)
                #     return loss + self.k * loss_dis
                #     return self.k * loss + (1-self.k)* loss_dis
                return loss

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