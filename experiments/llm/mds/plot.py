# %% Setup
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb

from mlem_method.viz import feature_rename, levels_rename

root = Path(__file__).parent
output = root.parent.parent.parent / "paper" / "figs" / "llm"
output.mkdir(parents=True, exist_ok=True)

mds = pd.read_parquet(root / "0.parquet")
mds.rename(columns=lambda col: feature_rename.get(col, col.split(".")[-1]), inplace=True)
mds = mds.replace(levels_rename)
mds["layer"] = pd.to_numeric(mds["layer"])

attachments, clauses, verbs = (
    ["Center embedding", "Peripheral"],
    ["Subject relative", "Object relative"],
    ["see", "like"],
)
# tab20 shade pairs of the llm_palette colors: attachment site blue, clause type orange, verb lemma green.
tab20 = sns.color_palette("tab20")
levels = dict(zip(attachments + clauses + verbs, tab20)) | {"Other": "0.85"}
dot = lambda color: plt.Line2D([], [], ls="", marker="o", color=color)
order = lambda data: data.sample(frac=1, random_state=0).sort_values("hue", key=lambda h: h != "Other", kind="stable")
kw = dict(x="coord_1", y="coord_2", hue="hue", edgecolor=None, rasterized=True, legend=False)
style = dict(xlabel="", ylabel="", xticks=[], yticks=[], aspect="equal")

# %% All layers, colored by clause type × attachment site
mds["hue"] = mds["Relative clause type"] + " × " + mds["Attachment site"].str.lower()
combos = dict(zip([f"{c} × {a.lower()}" for c in clauses for a in attachments], tab20))
g = sns.relplot(
    order(mds),
    palette=combos,
    col="layer",
    col_wrap=6,
    height=1.75,
    aspect=0.8,
    facet_kws={"sharex": False, "sharey": False},
    s=2,
    **kw,
)
g.set_titles("Layer {col_name}")
g.set(**style)
g.set_axis_labels("MDS 1", "MDS 2", alpha=0.5)
g.despine(left=True, bottom=True)
g.figure.subplots_adjust(wspace=0.05, hspace=0.35)
g.figure.legend(
    [dot(c) for c in combos.values()],
    combos,
    title="Relative clause type × Attachment site",
    loc="lower center",
    bbox_to_anchor=(0.5, 1),
    ncols=2,
    frameon=True,
    title_fontproperties={"weight": "bold"},
)
g.savefig(output / "mds_all_layers.pdf", dpi=200, metadata={"CreationDate": None})

# %% Layer-7 hierarchy: attachment site > verb lemma > clause type
l7 = mds.query("layer == 7")
A, V = l7["Attachment site"], l7["Verb lemma"]
features = {"Attachment site": attachments, "Verb lemma": verbs, "Relative clause type": clauses}
panels = {"A": ("Attachment site", A.notna(), "All\nsentences")}  # key: feature, branch, title
for a, key, leaves in zip(attachments, "BC", ["DE", "FG"]):
    panels[key] = ("Verb lemma", A == a, f"Attachment site:\n{a}")
    for v, leaf in zip(verbs, leaves):
        panels[leaf] = ("Relative clause type", (A == a) & (V == v), f'Verb lemma:\n"{v}"')

for name, paths in [("mds_hierarchy", ["ACG"]), ("mds_hierarchy_full", ["ABD", "ABE", "ACF", "ACG"])]:
    fig, axes = plt.subplots(
        len(paths), 3, figsize=(6, 2.4 * len(paths)), squeeze=False, gridspec_kw={"hspace": 0.45, "wspace": 0.05}
    )
    # seen = set()
    for row, path in zip(axes, paths):
        for ax, key in zip(row, path):
            # if key in seen:
            #     ax.set_axis_off()
            #     continue
            # seen.add(key)

            feature, branch, title = panels[key]
            sns.scatterplot(order(l7.assign(hue=l7[feature].where(branch, "Other"))), palette=levels, ax=ax, s=3, **kw)
            ax.set(title=title, **style)
            sns.despine(ax=ax, left=True, bottom=True)
    axes[-1, 0].set_xlabel("MDS 1", alpha=0.5)
    axes[-1, 0].set_ylabel("MDS 2", alpha=0.5)
    for i, (ax, (feature, values)) in enumerate(zip(axes[-1], features.items())):
        ax.legend(
            [dot(levels[v]) for v in values],
            values,
            title=f"{i + 1}. {feature}",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            frameon=False,
        )
    fig.savefig(output / f"{name}.pdf", dpi=300, metadata={"CreationDate": None})

# %%
