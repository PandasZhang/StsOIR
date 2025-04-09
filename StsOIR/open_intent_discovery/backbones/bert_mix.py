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

class BertMixNorm(BertPreTrainedModel):

    def __init__(self, config, args):

        super(BertMixNorm, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.lam = 0.9
        self.drop_prob = 0.3
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.activation_02 = activation_map["relu"]
        self.activation_03 = activation_map["softsign"]
        # self.dropout = nn.Dropout(self.drop_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        from torch.nn.utils import weight_norm
        self.norm = L2_normalization()
        self.classifier = weight_norm(
            nn.Linear(config.hidden_size, args.num_labels), name='weight')
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None, use_mix=False):
        # if mode != 'train':
        # if use_mix == False:
        #     return self.eval_forward(input_ids, token_type_ids, attention_mask, labels,
        #                              feature_ext, mode, loss_fct)

        encoded_layer_12, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        
        # meaned_output = encoded_layer_12[-1].mean(dim=1)

        pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
        # pooled_output = self.activation(pooled_output)
        pooled_output = self.activation_02(pooled_output)
        pooled_output = self.activation_03(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.norm(pooled_output)
        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            # tmp_key = 1
            if mode == 'train':
                # logits = self.softmax(logits)
                
                # labels_one_hot = to_one_hot(labels, self.num_labels)
                # labels_mixed = labels_one_hot * self.lam + labels_one_hot[indices] * (1 - self.lam)
                # cated_labels = torch.cat((labels_one_hot, labels_mixed), 0)
                # loss = loss_fct(logits, cated_labels)
                # logits_index = torch.max(logits, dim=-1)
                # loss_ori = loss_fct(logits[:logits.shape[0]//2], labels)
                # loss_mix = loss_fct(logits[logits.shape[0]//2:], labels)
                # loss = loss_ori * self.lam + (1-self.lam) * loss_mix

                loss = loss_fct(logits, labels)
                return loss
            else:
                return pooled_output, logits


class BertMix(BertPreTrainedModel):

    def __init__(self, config, args):

        super(BertMix, self).__init__(config)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        
        # bertcl部分
        # temp = 0.05
        self.dense_cl = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation_cl = activation_map[args.activation]
        # self.temp_cl = args.temp if args.temp is not None else 0.05
        self.temp_cl = 0.05
        self.cos_cl = nn.CosineSimilarity(dim=-1)
        self.loss_cl_fct = nn.CrossEntropyLoss()
        
        self.softmax = nn.Softmax(dim=1).cuda()
        self.lam = args.temp if args.temp is not None else 0.3
        # self.lam = 0.5
        # self.lam = 0.1
        
        
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None, use_mix=False):
        # if mode != 'train':
        if use_mix == False :
            return self.eval_forward(input_ids, token_type_ids, attention_mask, labels,
                                     feature_ext, mode, loss_fct)

        encoded_layer_12, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        last_hidden_state = encoded_layer_12[-1]
        # last_hidden_state = [batch_size, seq_len, hidden_size]
        # lam = get_lambda()
        # lam = 0.5
        labels_one_hot = None
        if labels is not None:
            labels_one_hot = to_one_hot(labels, self.num_labels)
        hidden_mixed, labels_mixed, indices = mixup_process(last_hidden_state, labels_one_hot, lam=self.lam)
        
        
        cated_hidden = torch.cat((last_hidden_state, hidden_mixed), 0)
        cated_labels = torch.cat((labels_one_hot, labels_mixed), 0)
        
        # # mean_ouput = [128, 768]
        mean_output = cated_hidden.mean(dim=1)
        mean_output = self.dense(mean_output)
        mean_output = self.activation(mean_output)
        pooled_output = self.dropout(mean_output)
        

        # logits_mixed = self.classifier(mixed_output)
        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            # tmp_key = 1
            if mode == 'train':
                # logits = self.softmax(logits)
                loss = self.loss_cl_fct(logits, cated_labels)

                # logits_mixed = self.softmax(logits_mixed)
                # loss_mixed = loss_fct(logits_mixed, labels_one_hot, labels_one_hot[indices])
                # return tmp_key * loss + (1-tmp_key) * loss_mixed
                # return tmp_key * loss_mixed
                return loss
            else:
                return pooled_output, logits
            
            
    def eval_forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None):
        ''' Pan
            因为加载dataloader是的batch_size设置了÷2,所以这里要先变换tensor形状到 batch_size*2, seq_len
        '''
        encoded_layer_12, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        last_hidden_state = encoded_layer_12[-1]
        # # mean_ouput = [128, 768]
        mean_output = last_hidden_state.mean(dim=1)
        mean_output = self.dense(mean_output)
        mean_output = self.activation(mean_output)
        mean_output = self.dropout(mean_output)
        
        logits = self.classifier(mean_output)

        if feature_ext:
            return mean_output
        else:
            if mode == 'train':
                loss = self.loss_cl_fct(logits, labels)
                return loss
            return mean_output, logits