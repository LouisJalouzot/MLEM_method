# MLEM Method

Implementation of the MLEM method for learning feature interactions.

## Setup

This project uses `uv` for dependency management.

1.  Install `uv`: Follow instructions at https://github.com/astral-sh/uv
2.  Create a virtual environment:
    ```bash
    uv venv
    ```
3.  Activate the environment:
    ```bash
    source .venv/bin/activate
    ```
4.  Install dependencies:
    Before `uv sync`, run the following command to install `torchsort` correctly:
    ```bash
    uv pip install --force-reinstall --no-cache-dir --no-build-isolation torchsort
    ```
    Then, sync the rest of the dependencies:
    ```bash
    uv sync
    ```

## Development

- The main entry points for running experiments are `main.py` and the Jupyter notebooks (`*.ipynb`).
- Core logic is located in `src/core/`.
- Infrastructure code (configuration, caching, data loading wrappers) is in `src/infra/`.
- Datasets are located in the `datasets/` directory.

## Testing

Unit tests are located in the `tests/` directory, mirroring the structure of `src/core/`.

To run the tests, use `pytest`:

```bash
pytest
```