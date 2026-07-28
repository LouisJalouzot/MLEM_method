python experiments/think_alike/long_range/feature_correlations.py

python experiments/think_alike/families/spearman.py \
    --input experiments/think_alike/long_range/1.parquet \
    --output-dir think_alike/figures/long_range

python experiments/think_alike/families/trajectories.py \
    --input experiments/think_alike/long_range/0.parquet \
    --output-dir think_alike/figures/long_range

python experiments/think_alike/families/dtw_figure.py \
    --input experiments/think_alike/long_range/0.parquet \
    --output-dir think_alike/figures/long_range