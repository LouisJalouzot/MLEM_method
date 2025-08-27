from setup_figs import EstimateCorrelations, clean_names, np, pd, plt, sns

df = clean_names(pd.read_csv("datasets/relative_clause.csv"))
for col in df.iloc[:, 1:]:
    print(col, ", ".join(df[col].unique().astype(str)))

ec = EstimateCorrelations(
    dataset={"path": "datasets/relative_clause.csv"}, product=False, device="cpu"
)
corrs = clean_names(ec.estimate_correlations()[0])
a = corrs.values
a[np.triu_indices_from(a)] = np.nan
corrs = pd.DataFrame(a[1:, :-1], index=corrs.index[1:], columns=corrs.columns[:-1])

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
)
colorbar = ax.collections[0].colorbar
ticks = colorbar.get_ticks()
colorbar.set_ticks(ticks[::2])
colorbar.set_label("Correlation", labelpad=20)
colorbar.ax.xaxis.set_label_position("top")
colorbar.ax.xaxis.set_ticks_position("top")
plt.xticks(rotation=45, ha="right")
plt.savefig("paper/figs/datasets/rc_correlations.pdf", bbox_inches="tight")
