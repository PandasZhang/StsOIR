
import os
import torch
import numpy as np
import pandas as pd
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME
from tqdm import tqdm

def save_npy(npy_file, path, file_name):
    npy_path = os.path.join(path, file_name)
    np.save(npy_path, npy_file)

def load_npy(path, file_name):
    npy_path = os.path.join(path, file_name)
    npy_file = np.load(npy_path)
    return npy_file

def save_model(model, model_dir):

    save_model = model.module if hasattr(model, 'module') else model  
    model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model_config_file = os.path.join(model_dir, CONFIG_NAME)
    torch.save(save_model.state_dict(), model_file)
    with open(model_config_file, "w") as f:
        f.write(save_model.config.to_json_string())

def restore_model(model, model_dir):
    output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model.load_state_dict(torch.load(output_model_file))
    return model

def save_results(args, test_results):

    pred_labels_path = os.path.join(args.method_output_dir, 'y_pred.npy')
    np.save(pred_labels_path, test_results['y_pred'])
    true_labels_path = os.path.join(args.method_output_dir, 'y_true.npy')
    np.save(true_labels_path, test_results['y_true'])

    del test_results['y_pred']
    del test_results['y_true']

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    import datetime
    created_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    var = [args.dataset, args.method, args.backbone, args.known_cls_ratio, args.labeled_ratio, args.loss_fct, args.seed,args.tmp_k, args.num_train_epochs, created_time]
    names = ['dataset', 'method', 'backbone', 'known_cls_ratio', 'labeled_ratio', 'loss', 'seed', 'tmp_k', 'train_epochs', 'created_time']
    vars_dict = {k:v for k,v in zip(names, var) }
    results = dict(test_results,**vars_dict)
    keys = list(results.keys())
    values = list(results.values())
    
    results_path = os.path.join(args.result_dir, args.results_file_name)
    
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori,columns = keys)
        df1.to_csv(results_path,index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results,index=[1])
        df1 = df1.append(new,ignore_index=True)
        df1.to_csv(results_path,index=False)
    data_diagram = pd.read_csv(results_path)
    
    print('test_results', data_diagram)


def centroids_cal(model, args, data, train_dataloader, device):
    
    model.eval()
    centroids = torch.zeros(data.num_labels, args.feat_dim).to(device)
    delta = torch.zeros(data.num_labels).to(device)
    total_labels = torch.empty(0, dtype=torch.long).to(device)
    total_features = torch.empty((0,args.feat_dim)).to(device)
    with torch.set_grad_enabled(False):

        for batch in tqdm(train_dataloader, desc="Calculate centroids"):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            features = model(input_ids, segment_ids, input_mask, feature_ext=True)
            total_labels = torch.cat((total_labels, label_ids))
            total_features = torch.cat( (total_features, features) )
            for i in range(len(label_ids)):
                label = label_ids[i]
                centroids[label] += features[i]

    total_labels = total_labels.cpu().numpy()
    centroids /= torch.tensor(class_count(total_labels)).float().unsqueeze(1).to(device)


    d = centroids[total_labels]
    dis = torch.norm( total_features - d, 2, 1 ).view(-1)
    for label_i in range(data.num_labels):
        # index = total_labels == label_i
        delta[label_i] = dis[total_labels == label_i].mean()
    return centroids

def centroids_deltas_cal(model, args, data, train_dataloader, device):
    
    model.eval()
    centroids = torch.zeros(data.num_labels, args.feat_dim).to(device)
    delta = torch.zeros(data.num_labels).to(device)
    total_labels = torch.empty(0, dtype=torch.long).to(device)
    total_features = torch.empty((0,args.feat_dim)).to(device)
    with torch.set_grad_enabled(False):

        for batch in tqdm(train_dataloader, desc="Calculate centroids"):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            features = model(input_ids, segment_ids, input_mask, feature_ext=True)
            total_labels = torch.cat((total_labels, label_ids))
            total_features = torch.cat( (total_features, features) )
            for i in range(len(label_ids)):
                label = label_ids[i]
                centroids[label] += features[i]

    total_labels = total_labels.cpu().numpy()
    centroids /= torch.tensor(class_count(total_labels)).float().unsqueeze(1).to(device)


    d = centroids[total_labels]
    dis = torch.norm( total_features - d, 2, 1 ).view(-1)
    for label_i in range(data.num_labels):
        # index = total_labels == label_i
        delta[label_i] = dis[total_labels == label_i].mean()
    return centroids, delta


def save_train_features(model, args, data, train_dataloader, device):
    
    model.eval()
    centroids = torch.zeros(data.num_labels, args.feat_dim).to(device)
    delta = torch.zeros(data.num_labels).to(device)
    total_labels = torch.empty(0, dtype=torch.long).to(device)
    total_features = torch.empty((0,args.feat_dim)).to(device)
    with torch.set_grad_enabled(False):

        for batch in tqdm(train_dataloader, desc="Calculate centroids"):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            features = model(input_ids, segment_ids, input_mask, feature_ext=True)
            total_labels = torch.cat((total_labels, label_ids))
            total_features = torch.cat( (total_features, features) )
            for i in range(len(label_ids)):
                label = label_ids[i]
                centroids[label] += features[i]

    # total_labels = total_labels.cpu().numpy()
    # total_features = total_features.cpu().numpy()

    np.save(os.path.join(args.method_output_dir, 'train_features.npy'), total_features.detach().cpu().numpy())
    np.save(os.path.join(args.method_output_dir, 'train_labels.npy'), total_labels.detach().cpu().numpy())
    # centroids /= torch.tensor(class_count(total_labels)).float().unsqueeze(1).to(device)


    # d = centroids[total_labels]
    # dis = torch.norm( total_features - d, 2, 1 ).view(-1)
    # for label_i in range(data.num_labels):
    #     # index = total_labels == label_i
    #     delta[label_i] = dis[total_labels == label_i].mean()
    # return centroids, delta


def class_count(labels):
    class_data_num = []
    for l in np.unique(labels):
        num = len(labels[labels == l])
        class_data_num.append(num)
    return class_data_num


#https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py#L70
def mask_tokens(inputs, tokenizer,\
    special_tokens_mask=None, mlm_probability=0.15):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        # shape与inputs相同，值全为mlm_probability
        probability_matrix = torch.full(labels.shape, mlm_probability)
        
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs==0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

