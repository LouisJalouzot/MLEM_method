# MLEM Method

Implementation of the MLEM method for learning feature interactions.

## Setup

This project uses `uv` for dependency management.

1.  Install `uv`:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    You may need to restart your terminal.
2.  Create a virtual environment:
    ```bash
    uv venv .venv --python=3.13
    ```
3.  Activate the environment:
    ```bash
    source .venv/bin/activate
    ```
4.  Install dependencies (`torchsort` is a bit finicky, so we install and build it separately):
    Prepare the environment for `torchsort`:
    ```bash
    uv pip install setuptools "torch>=2.6.0"
    ```
    Build and install `torchsort`:
    ```bash
    uv pip install --no-build-isolation torchsort
    ```
    Install the rest of the dependencies:
    ```bash
    uv sync
    ```

**Everything together:**
```bash
uv venv .venv --python=3.13
source .venv/bin/activate
uv pip install setuptools "torch>=2.6.0"
uv pip install --no-build-isolation torchsort
uv sync
```