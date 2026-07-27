from argparse import ArgumentParser
from pathlib import Path

from mlem_method import EstimateCorrelations
from mlem_method.viz import *

parser = ArgumentParser()
parser.add_argument("--dataset", default="datasets/long_range_agreement.csv")
parser.add_argument("--output-dir", type=Path, default=Path("think_alike/figures/long_range"))
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

correlations, n_pairs = EstimateCorrelations(
    dataset={"path": args.dataset}, product=False, device="cpu"
).estimate_correlations()
correlations.to_csv(Path(__file__).parent / "feature_correlations.csv")
correlations = correlations.rename(columns=feature_rename, index=feature_rename)

values = correlations.to_numpy(copy=True)
np.fill_diagonal(values, 0)
print(f"pairs={n_pairs}, max |r|={np.abs(values).max():.3f}")

values[np.triu_indices_from(values)] = np.nan
correlations = pd.DataFrame(values[1:, :-1], index=correlations.index[1:], columns=correlations.columns[:-1])

fig, ax = plt.subplots(figsize=(6, 6))
annotations = correlations.round(2).where(correlations.abs() > 0.4, "")
sns.heatmap(
    correlations,
    cmap="RdBu_r",
    annot=annotations,
    fmt="",
    vmin=-1,
    vmax=1,
    cbar_kws={"orientation": "horizontal", "location": "top"},
    square=True,
    ax=ax,
)
colorbar = ax.collections[0].colorbar
ticks = colorbar.get_ticks()
colorbar.set_ticks(ticks[::2])
colorbar.set_label("Correlation", labelpad=10)
colorbar.ax.xaxis.set_label_position("top")
colorbar.ax.xaxis.set_ticks_position("top")
plt.xticks(rotation=45, ha="right")
plt.savefig(args.output_dir / "feature_correlations.pdf", bbox_inches="tight")
