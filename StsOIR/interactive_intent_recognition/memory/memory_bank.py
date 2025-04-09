import torch
from sklearn.cluster import KMeans

class MemoryBank:
    def __init__(self):
        self.memory = {"embeddings": [], "labels": []}

    def update(self, z, labels):
        self.memory["embeddings"].append(z.detach().cpu())
        self.memory["labels"].append(labels.detach().cpu())

    def sample_prototypes(self, n_per_class=5):
        embs = torch.cat(self.memory["embeddings"])
        labels = torch.cat(self.memory["labels"])
        selected_idxs = []
        for c in torch.unique(labels):
            indices = (labels == c).nonzero(as_tuple=True)[0]
            if len(indices) <= n_per_class:
                selected_idxs.extend(indices.tolist())
            else:
                kmeans = KMeans(n_clusters=n_per_class).fit(embs[indices])
                centers = torch.tensor(kmeans.cluster_centers_)
                dists = torch.cdist(embs[indices], centers)
                selected = indices[torch.argmin(dists, dim=0)]
                selected_idxs.extend(selected.tolist())
        return embs[selected_idxs], labels[selected_idxs]
