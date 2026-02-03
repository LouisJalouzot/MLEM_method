# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

from mlem import SentenceRepresentations

os.chdir("../")


# %%
def flatten_triu(M):
    return M[np.triu_indices(len(M), k=1)]


def make_random_rdm(n, d=768):
    vectors = np.random.randn(n, d)
    return np.sqrt(((vectors[:, None] - vectors[None, :]) ** 2).sum(axis=-1))


repr = SentenceRepresentations()
Y = repr()
rdm = np.sqrt(((Y[:, None] - Y[None, :]) ** 2).sum(axis=-1))
rdm = flatten_triu(rdm)


pearson, spearman = [], []
for _ in tqdm(range(128)):
    rand_rdm = flatten_triu(make_random_rdm(Y.shape[0]))
    pearson.append(pearsonr(rdm, rand_rdm)[0])
    spearman.append(spearmanr(rdm, rand_rdm)[0])
pearson = np.array(pearson)
spearman = np.array(spearman)


# %%
plt.hist(pearson, bins=20, alpha=0.7, label="Pearson", color="steelblue")
plt.hist(spearman, bins=20, alpha=0.7, label="Spearman", color="coral")

plt.show()

# %%
pearson, spearman = [], []
for _ in tqdm(range(128)):
    Y_perm = np.random.permutation(Y)
    rdm_perm = np.sqrt(((Y_perm[:, None] - Y_perm[None, :]) ** 2).sum(axis=-1))
    rdm_perm = flatten_triu(rdm_perm)
    pearson.append(pearsonr(rdm, rdm_perm)[0])
    spearman.append(spearmanr(rdm, rdm_perm)[0])
pearson = np.array(pearson)
spearman = np.array(spearman)

# %%
plt.hist(pearson, bins=20, alpha=0.7, label="Pearson", color="steelblue")
plt.hist(spearman, bins=20, alpha=0.7, label="Spearman", color="coral")

plt.show()
