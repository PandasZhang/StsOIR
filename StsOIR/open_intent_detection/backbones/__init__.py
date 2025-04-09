from .bert import BERT, BERT_Norm
from .bert_mix import BertMix
from .bert_rnn import BERTRNN
from .bert_suploss import BERT_SUP, BERT_SUP_Norm
from .bert_knn import BERT_Knn, BERT_Knn_Norm
from .bert_mtp import BertMtp
backbones_map = {
                    'bert': BERT, 
                    'bert_norm': BERT_Norm,
                    'bert_mix': BertMix,
                    'bert_rnn': BERTRNN,
                    'bert_mtp': BertMtp,
                    'bert_sup': BERT_SUP,
                    'bert_sup_norm': BERT_SUP_Norm,
                    'bert_knn': BERT_Knn,
                    'bert_knn_norm': BERT_Knn_Norm,
                }