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


class BertCL(BertPreTrainedModel):

    def __init__(self, config, args):

        super(BertCL, self).__init__(config)
        self.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')   
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]

        self.hidden_dropout_prob = 0.1
        self.feat_dim = 128


        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)      

        
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.feat_dim)
        )
        self.backbone.to(self.device)
        self.head.to(self.device)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = False)
        pooled_output = self.dense(encoded_layer_12.mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                
                loss = loss_fct(logits, labels)
                return loss
            else:
                return pooled_output, logits
            
            