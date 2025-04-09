from unicodedata import bidirectional
import torch
import math
import torch.nn.functional as F
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch.nn.parameter import Parameter
from .utils import L2_normalization

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'elu': nn.ELU()}

class BERTRNN(BertPreTrainedModel):

    def __init__(self, config, args):

        super(BERTRNN, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        '''
        bigru部分
        '''
        # self.rnn = nn.GRU(input_size=config.hidden_size, hidden_size=args.train_batch_size,
        #                     num_layers=self.num_layers,
        #                     batch_first=True, bidirectional=True)

        '''
        SCLoss部分所需
        '''
        self.norm_coef = 0.1



        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None):
        '''
        拆分为两部分：
            第一部分：作为基础的特征抽取器，进行原始的特征抽取工作，不进行额外的计算。
            第二部分：对样本进行增强，并采用SCLoss进行backwards传播。
        '''
        encoded_layer_12, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        '''     以上为特征抽取器部分    '''
        meaned_pooled_output = encoded_layer_12[-1].mean(dim=1)
        seq_embed = meaned_pooled_output.clone().detach().requires_grad_(True).float()
        '''     以上为特征转换部分    '''
        pooled_output = self.dense(seq_embed)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        '''     以上为原始logits计算部分    '''
        if feature_ext:
            return pooled_output
        '''     以上为原始特征抽取部分    '''
        if mode != 'train':
            return pooled_output, logits
        '''     以上为原始logtis计算部分    '''
        loss = loss_fct(logits, labels)
        seq_embed.retain_grad()
        unnormalized_noise = seq_embed.grad.detach_()
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        norm = unnormalized_noise.norm(p=2,dim=-2)
        normalized_noize = unnormalized_noise / (norm.unsqueeze(dim=-1) + 1e-10)
        noise_embedding = seq_embed + self.norm_coef * normalized_noize
        '''     以上为noise_embedding计算部分    '''
        _pooled_output = self.dense(noise_embedding)
        _pooled_output = self.activation(_pooled_output)
        _pooled_output = self.dropout(_pooled_output)
        '''     以上为embedding再过一次密集层部分    '''
        label = F.one_hot(labels, num_classes=self.num_labels).cpu()
        label_mask = torch.mm(label,label.T).bool().long().cuda()
        sup_cont_loss = nt_xent(pooled_output, _pooled_output, label_mask, cuda=self.use_cuda=='cuda')
        '''     以上为SCLoss计算部分    '''
        return sup_cont_loss