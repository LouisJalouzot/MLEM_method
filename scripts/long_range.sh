python scripts/feature_correlations.py \
    --dataset datasets/long_range_agreement_2.csv \
    --output-dir think_alike/figures/long_range

python experiments/think_alike/families/spearman.py --long-range
python experiments/think_alike/families/feature_importance.py --long-range
python experiments/think_alike/families/trajectories.py --long-range
python experiments/think_alike/dtw/fi_figure.py --long-range
python experiments/think_alike/dtw/mds_figure.py --long-range
python experiments/think_alike/dtw/metadata_figures.py