from mlem import EstimateCorrelations, SyntMov2024Dataset

from setup_figs import np, pd, plt, sns

ec = EstimateCorrelations(dataset=SyntMov2024Dataset(), product=False, device="cpu")
corrs = ec.estimate_correlations()[0]
a = corrs.values.copy()
a[np.triu_indices_from(a)] = np.nan
corrs = pd.DataFrame(a[1:, :-1], index=corrs.index[1:], columns=corrs.columns[:-1])

# Create annotation mask for correlations with absolute value > 0.3
annot_data = corrs.copy()
annot_data[np.abs(corrs) <= 0.3] = ""
annot_data[np.abs(corrs) > 0.3] = corrs[np.abs(corrs) > 0.3].round(2).astype(str)

plt.figure(figsize=(10, 10))
ax = sns.heatmap(
    corrs,
    cmap="RdBu",
    vmin=-1,
    vmax=1,
    cbar_kws={
        "orientation": "horizontal",
        "location": "top",
    },
    square=True,
    annot=annot_data,
    fmt="",
    annot_kws={"fontsize": 14},
)
colorbar = ax.collections[0].colorbar
ticks = colorbar.get_ticks()
colorbar.set_ticks(ticks[::2])
colorbar.set_label("Correlation", labelpad=20)
colorbar.ax.xaxis.set_label_position("top")
colorbar.ax.xaxis.set_ticks_position("top")
plt.xticks(rotation=45, ha="right")
plt.savefig("paper/figs/datasets/syntmov2024_correlations.pdf", metadata={"CreationDate": None})
