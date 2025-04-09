import torch
import torch.nn as nn
from transformers import BertModel

class InteractiveBERTEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_size=768, num_labels=77):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5
        self.dropout = nn.Dropout(0.1)

        # 分类器
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        z = output.last_hidden_state  # [B, L, H]

        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        z_new = torch.matmul(weights, v)

        z_mean = z_new.mean(dim=1)  # [B, H] 句子级别表示
        logits = self.classifier(z_mean)

        return logits, z_mean  # 分类结果 和 表征向量
