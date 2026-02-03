import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


def flatten_triu(M):
    return M[np.triu_indices(len(M), k=1)]


def make_random_rdm(n, d=768):
    vectors = np.random.uniform(size=(n, d))
    return np.sqrt(((vectors[:, None] - vectors[None, :]) ** 2).sum(axis=-1))


def compute_correlations(n, n_pairs=500, pbar=None):
    pearson, spearman = [], []
    for _ in range(n_pairs):
        rdm1 = make_random_rdm(n)
        rdm2 = make_random_rdm(n)
        x, y = flatten_triu(rdm1), flatten_triu(rdm2)
        if pbar:
            pbar.update(1)
        pearson.append(pearsonr(x, y)[0])
        spearman.append(spearmanr(x, y)[0])
    return np.array(pearson), np.array(spearman)


sizes = [64, 128, 256]
size_pairs = {64: 256, 128: 256, 256: 256}
fig, axes = plt.subplots(1, len(sizes), figsize=(14, 3.5), sharey=True)

total_iters = sum(size_pairs.values())
with tqdm(total=total_iters, desc="Computing correlations") as pbar:
    for ax, n in zip(axes, sizes):
        p, s = compute_correlations(n, size_pairs[n], pbar)
        ax.hist(p, bins=20, alpha=0.7, label="Pearson", color="steelblue")
        ax.hist(s, bins=20, alpha=0.7, label="Spearman", color="coral")
        ax.set_title(f"Number of stimuli: {n}", fontsize=12)
        ax.set_xlabel("Correlation", fontsize=10)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        if ax == axes[0]:
            ax.set_ylabel("Count", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

plt.suptitle(
    "Distribution of Correlations Between RDMs of Random Vectors", fontsize=14, y=1.02
)
plt.tight_layout()
plt.savefig("rdm_correlations.png", dpi=150, bbox_inches="tight")
plt.show()
