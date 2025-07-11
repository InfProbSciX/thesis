
import gc, torch
import numpy as np
from tqdm import trange
from pykeops.torch import LazyTensor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt; plt.ion()

np.random.seed(42); torch.manual_seed(42)

x, y = torch.load('mnist.pth', weights_only=False)
x = x[:10000]
y = y[:10000]
n = len(x)

k = 15
init = PCA(n_components=2).fit_transform(x)
init /= (init[:, 0].std())

model = torch.nn.Embedding.from_pretrained(torch.tensor(init), freeze=False).to("cuda")

x_cuda = torch.tensor(x).to("cuda").contiguous()
dists = (LazyTensor(x_cuda[:, None]) - LazyTensor(x_cuda[None])).square().sum(-1)
knn_idx = dists.argKmin(K=k + 1, dim=0)[:, 1:].cpu() # .numpy().flatten()
del x_cuda, dists

r, c = torch.triu_indices(n, n, 1)

nn = torch.zeros(n, n).to("cuda")
nn[torch.arange(n)[:, None], knn_idx] = 1
A = (nn + nn.T).clip(0, 1)

n_bar = 2*15*5/1.5
eps = n_bar / n

I = torch.eye(n).to("cuda")
O = torch.ones(n, n).to("cuda")
H = I - O/n
L = A.sum(0).diag() - A

def bound_interp():
    X = model.weight # - model.weight.mean(0)
    D = (X[:, None] - X[None]).square().sum(-1)
    p = 1/(1 + D)
    Si = I/(2*eps) + 0.5*(H @ p @ H) + X@X.T
    return (L @ Si).trace() - n*Si.logdet()

torch.manual_seed(0)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0, lr=1.0)
for epoch in (bar := trange(200, leave=False)):
    optimizer.param_groups[0]["lr"] = 1 - epoch/200
    model.train()
    loss = bound_interp()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    bar.set_description(f"L:{loss.item()}")
    gc.collect(); torch.cuda.empty_cache()

embd_neg = model.weight.detach().cpu().numpy()

plt.scatter(*embd_neg.T, c=y, alpha=0.75, s=1, cmap="tab10", edgecolor="none")
plt.axis('off')
plt.tight_layout()

# pca

plt.scatter(*init.T, c=y, alpha=0.75, s=1, cmap="tab10", edgecolor="none")
plt.axis('off')
plt.tight_layout()

# gplvm

x_std = torch.tensor(x / x.std()).to("cuda")

I = torch.eye(n).to("cuda")
O = torch.ones(n, n).to("cuda")
mu = x_std.mean(axis=0)[None, :]
model = torch.nn.Embedding.from_pretrained(torch.tensor(init), freeze=False).to("cuda")
log_p = torch.nn.Parameter(torch.ones(4).to("cuda"))
torch.manual_seed(0)
optimizer = torch.optim.Adam(list(model.parameters()) + [log_p], weight_decay=0.0, lr=0.025)
for epoch in (bar := trange(300, leave=False)):
    model.train()

    X = model.weight - model.weight.mean(0)
    D = (X[:, None] - X[None]).square().sum(-1)

    p_o, p_l, p_r, p_n = log_p.sigmoid()

    if epoch <= 10:
        X = X.detach()
        D = D.detach()

    loss = -torch.distributions.MultivariateNormal(O[[0], :], p_l*(X @ X.T) + p_r/(1 + D) + p_o*O + p_n*I).log_prob(x_std.T).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    bar.set_description(f"L:{loss.item()}")
    gc.collect(); torch.cuda.empty_cache()

embd_neg = model.weight.detach().cpu().numpy()

plt.scatter(*embd_neg.T, c=y, alpha=0.75, s=1, cmap="tab10", edgecolor="none")
plt.axis('off')
plt.tight_layout()
