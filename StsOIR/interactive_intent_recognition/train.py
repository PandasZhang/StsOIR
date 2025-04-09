# 替换原模型调用部分
logits, z = model(input_ids, attn_mask)

# 分类损失
loss_cls = nn.CrossEntropyLoss()(logits, labels)

# 对比损失
loss_contrast = contrastive_loss(z, labels, config["temperature"])

# 知识蒸馏
if len(memory.memory["embeddings"]) > 0:
    old_embs, old_labels = memory.sample_prototypes()
    old_embs, old_labels = old_embs.to(device), old_labels.to(device)
    P = compute_prototypes(old_embs, old_labels)
    Q = compute_prototypes(z, labels)
    loss_kd = prototype_kl(P, Q)
else:
    loss_kd = torch.tensor(0.0).to(device)

# 总损失
loss = loss_cls + loss_contrast + loss_kd
