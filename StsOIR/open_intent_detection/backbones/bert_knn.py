from turtle import forward
import torch
import math
import torch.nn.functional as F
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch.nn.parameter import Parameter
from .utils import L2_normalization

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'elu': nn.ELU()}

#remote_sever
def l2norm(x: torch.Tensor):
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    x = torch.div(x, norm)
    return x

class BERT_Knn(BertPreTrainedModel):

    def __init__(self, config, args):

        super(BERT_Knn, self).__init__(config)
        self.num_labels = args.num_labels
        # TODO  复现Knn，构建 bert_k 与 bert_q ，以及几个作用不同的分类头
        # 复现 knn方法
        self.bert_k = BertModel(config)
        self.bert_q = BertModel(config)
        self.contrastive_liner_q = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.contrastive_liner_k = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.classifier_liner = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.num_labels)
        )
        # self.classifier_liner = BertClassificationHead(config, self.num_labels)

        # super param
        self.m = 0.999  # be use to update bert_k bert_q 
        self.T = args.temperature   # 计算对比逻辑损失时的分母

        # create the label_queue and feature_queue
        self.K = args.queue_size # 7500

        self.register_buffer("label_queue", torch.randint(0, self.number_labels, [self.K])) # Tensor:(7500,)
        self.register_buffer("feature_queue", torch.randn(self.K, config.hidden_size)) # Tensor:(7500, 768)
        self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long)) # Tensor(1,)
        self.top_k = args.knn_num # 25
        self.update_num = args.positive_num # 3

        # optional and delete can improve the performance indicated 
        # by some experiment
        params_to_train = ["layer." + str(i) for i in range(0,12)]
        for name, param in self.encoder_q.named_parameters():
            param.requires_grad_(False)
            for term in params_to_train:
                if term in name:
                    param.requires_grad_(True)


        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self.init_bert_weights)

    
    def init_weights(self):
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_q.data
    def save_negative_dataset(self, train_dataset):
        negative_dataset = {}
        data = train_dataset
        for line in data:
            print()
    
    def update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    def reshape_dict(self, batch):
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, shape[-1]])
        return batch
        
    def _dequeue_and_enqueue(self, keys, label):
        # keys = concat_all_gather(keys)
        # label = concat_all_gather(label)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            keys = keys[: batch_size]
            label = label[: batch_size]

        # replace the keys at ptr (dequeue ans enqueue)
        self.feature_queue[ptr: ptr + batch_size, :] = keys
        self.label_queue[ptr: ptr + batch_size] = label

        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    
    def select_pos_neg_sample(self, liner_q: torch.Tensor, label_q: torch.Tensor):
        print(torch.cuda.is_available())
        label_queue = self.label_queue.clone().detach()        # K
        feature_queue = self.feature_queue.clone().detach()    # K * hidden_size

        # 1. expand label_queue and feature_queue to batch_size * K
        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])   # tmp_label_queue = [batch_size, K]
        tmp_feature_queue = feature_queue.unsqueeze(0)          # 1 * K * hidden_size
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1]) # batch_size * K * hidden_size

        # 2.caluate sim, 计算
        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        # 3. get index of postive and neigative 
        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])       # batch_size, K
        # 对比 input 的 真实标签 tmp_label 与 对比序列的标签 tmp_label_queue
        pos_mask_index = torch.eq(tmp_label_queue, tmp_label)
        neg_mask_index = ~ pos_mask_index

        # 4.another option
        feature_value = cos_sim.masked_select(pos_mask_index)
        pos_sample = torch.full_like(cos_sim, -np.inf)
        # pos_sample = torch.full_like(cos_sim, -np.inf).cuda()
        #pos_sample = pos_sample.masked_scatter(pos_mask_index, feature_value)

        feature_value = cos_sim.masked_select(neg_mask_index)
        # torch.cuda.current_device()
        # torch.cuda._initialized = True
        neg_sample = torch.full_like(cos_sim, -np.inf)
        # neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        # 5.topk
        pos_mask_index = pos_mask_index.int()       # pos_mask_index = [batch_size, K]
        pos_number = pos_mask_index.sum(dim=-1)     # pos_number = [batch_size]
        pos_min = pos_number.min()
        if pos_min == 0:
            return None
        pos_sample, _ = cos_sim.topk(pos_min, dim=-1)   # cos_sim = [batch_size, K], pos_sample = [batch_size, pos_min]
        #pos_sample, _ = pos_sample.topk(pos_min, dim=-1)
        pos_sample_top_k = pos_sample[:, 0:self.top_k] # self.topk = 25
        #pos_sample_top_k = pos_sample[:, 0:self.top_k]
        #pos_sample_last = pos_sample[:, -1]
        ##pos_sample_last = pos_sample_last.view([-1, 1])
        pos_sample = pos_sample_top_k                   # pos_sample = [batch_size, topk]
        #pos_sample = torch.cat([pos_sample_top_k, pos_sample_last], dim=-1)
        pos_sample = pos_sample.contiguous().view([-1, 1])

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_topk = min(pos_min, self.top_k)
        neg_sample = neg_sample.repeat([1, neg_topk])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.T
        return logits_con

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None):

        # 使用 positive_sample 经过 bert_k， 
        # 使用contrastive_liner_k 获取tmp_labels ，
        # 后更新_dequeue_and_enqueue
        with torch.no_grad():
            self.update_encoder_k()
        # 使用inputs 经过 bert_q ，
        # 使用 contrastive_liner_q 与 classifier_liner 得到经典logits_cls
        # 使用交叉熵获取分数


        # 调用select_pos_neg_sample方法获取logits_con
        # 调用交叉熵获取分数

        # 对两个分数应用权重相加得到总分数
        return 0

    def update_bert_k_queue(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        with torch.no_grad():
            self.update_encoder_k()
            encoded_layer_12, pooled_output  = self.bert_k(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
            pooled_output = encoded_layer_12[-1].mean(dim=1)
            pooled_output = self.contrastive_liner_k(pooled_output)
            pooled_output = l2norm(pooled_output)
            tmp_labels = labels.unsqueeze(-1)
            tmp_labels = tmp_labels.repeat([1, self.update_num])
            tmp_labels = tmp_labels.view(-1)
            self._dequeue_and_enqueue(pooled_output, tmp_labels)



    def forward_ori(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None):

        encoded_layer_12, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss = loss_fct(logits, labels)
                # loss_cel = self.loss_fct(logits, labels)
                # return loss + loss_cel
                return loss
            else:
                return pooled_output, logits




class BERT_Knn_Norm(BertPreTrainedModel):

    def __init__(self, config, args):

        super(BERT_Knn_Norm, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        from torch.nn.utils import weight_norm
        self.norm = L2_normalization()
        self.classifier = weight_norm(
            nn.Linear(config.hidden_size, args.num_labels), name='weight')
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None):

        encoded_layer_12, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.norm(pooled_output)
        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss = loss_fct(logits, labels)
                return loss
            else:
                return pooled_output, logits
