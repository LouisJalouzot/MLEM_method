import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import make_sparse_spd_matrix, make_spd_matrix
from tqdm import tqdm


def flatten_triu(M):
    return M[np.triu_indices(len(M), k=1)]


def compute_correlations(n, n_pairs=500, pbar=None):
    pearson, spearman = [], []
    for _ in range(n_pairs):
        x, y = (
            flatten_triu(make_spd_matrix(n)),
            flatten_triu(make_spd_matrix(n)),
        )
        if pbar:
            pbar.update(1)
        pearson.append(pearsonr(x, y)[0])
        spearman.append(spearmanr(x, y)[0])
    return np.array(pearson), np.array(spearman)


sizes = [64, 128, 256]
size_pairs = {64: 128, 128: 128, 256: 128}
fig, axes = plt.subplots(1, len(sizes), figsize=(14, 3.5), sharey=True)

total_iters = sum(size_pairs.values())
with tqdm(total=total_iters, desc="Computing correlations") as pbar:
    for ax, n in zip(axes, sizes):
        p, s = compute_correlations(n, size_pairs[n], pbar)
        ax.hist(p, bins=20, alpha=0.7, label="Pearson", color="steelblue")
        ax.hist(s, bins=20, alpha=0.7, label="Spearman", color="coral")
        ax.set_title(f"Matrix size: {n}×{n}", fontsize=12)
        ax.set_xlabel("Correlation", fontsize=10)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        if ax == axes[0]:
            ax.set_ylabel("Count", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

plt.suptitle(
    "Distribution of Correlations Between Random SPD Matrices", fontsize=14, y=1.02
)
plt.tight_layout()
plt.savefig("spd_correlations.png", dpi=150, metadata={"CreationDate": None})
plt.show()
