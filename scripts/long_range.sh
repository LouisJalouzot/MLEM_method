python scripts/feature_correlations.py \
    --dataset datasets/long_range_agreement_2.csv \
    --output-dir think_alike/figures/long_range_2

python experiments/think_alike/families/spearman.py \
    --input experiments/think_alike/long_range_2/1.parquet \
    --output-dir think_alike/figures/long_range_2

python experiments/think_alike/families/feature_importance.py \
    --input experiments/think_alike/long_range_2/0.parquet \
    --output-dir think_alike/figures/long_range_2

python experiments/think_alike/families/trajectories.py \
    --input experiments/think_alike/long_range_2/0.parquet \
    --output-dir think_alike/figures/long_range_2

python experiments/think_alike/dtw/fi_figure.py --long-range
python experiments/think_alike/dtw/mds_figure.py --long-range
python experiments/think_alike/dtw/metadata_figures.py