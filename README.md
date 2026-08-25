# MLEM Method

Implementation of the MLEM method for learning feature interactions.

## Setup

This project uses `uv` for dependency management.

1.  Install `uv`:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    You may need to restart your terminal.

2.  Create and activate a virtual environment with required dependencies:
    ```bash
    uv sync
    source .venv/bin/activate
    ```

## Mahalanobis MLEM

Enable full Mahalanobis learning with one dataset flag:

```yaml
dataset:
  mahalanobis: true
pfi_grouping: feature  # or coordinate
```

Continuous features keep the legacy min-max scaling. A categorical feature with
`C` levels becomes `C - 1` unit-simplex (Helmert) coordinates. Pairwise input is
the signed coordinate difference, so the existing SPD learner fits
`delta.T @ W @ delta`.

`Dataset.features` and `Dataset.pfeatures` describe theoretical features and
their interactions. `Dataset.coordinates` and `Dataset.pcoordinates` describe
the model input and the upper triangle of `W`. `Dataset.encode()` returns
`(X, groups)`, where `groups` maps each coordinate to its theoretical feature;
`Dataset.pcoordinate_groups` gives the corresponding mapping for quadratic
coordinate terms.

`pfi_grouping: feature` jointly permutes all coordinate terms belonging to one
theoretical feature or interaction. `pfi_grouping: coordinate` reports each
Helmert coordinate or coordinate pair separately while retaining its `Group`.

## THINGS-data

Download fMRI betas, MEG epochs and concept annotations into `data/things`
(~80 GB total; the fMRI step additionally needs ~43 GB of temporary space):

```bash
./scripts/download_things.sh annotations
./scripts/download_things.sh fmri --dest data/things 01 02 03
./scripts/download_things.sh meg --dest data/things 01 02 03 04
```

Layout:

```text
data/things/annotations/property-ratings.tsv      # THINGSplus concept features (X)
data/things/fmri/betas_csv/sub-*_ResponseData.h5   # single-trial betas (Y)
data/things/fmri/betas_csv/sub-*_{Stimulus,Voxel}Metadata.csv
data/things/meg/preprocessed_P*-epo*.fif           # preprocessed epochs (Y)
```

Quick sanity check without big downloads: `./scripts/download_things.sh tiny`.