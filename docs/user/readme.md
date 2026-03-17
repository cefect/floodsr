# USER documentation

- [ReadTheDocs](https://app.readthedocs.org/projects/floodsr/)

## Read the Docs config

- `.readthedocs.yaml`

## Local build with Sphinx

```bash
# 1) go to the docs source directory
cd /workspace/docs/user

# 2) verify sphinx is available in the current environment
python -m sphinx --version

# 3) build html docs from this directory into BUILD_DIR/html
python -m sphinx -b html . "_build/manual"

# launch index.html in the default Windows browser (from WSL)
"\\wsl.localhost\Ubuntu\home\cefect\LS\09_REPOS\04_TOOLS\floodsr\docs\user\_build\manual\index.html"
```

## update tutorial notebooks

```bash
# build the preconfigured notebook runner image once
export IMAGE_NAME="cefect/floodsr:tutorial-notebooks-v0.1"
docker buildx build --load -t "${IMAGE_NAME}" -f container/tutorial_notebooks/Dockerfile .

# run from the repository root; notebook outputs are written back into the
# .ipynb files and leftover docs/user/notebooks/*.tif files are removed at the end
cd /home/cefect/LS/09_REPOS/04_TOOLS/floodsr
FLOODSR_NOTEBOOK_IMAGE="${IMAGE_NAME}" bash docs/user/scripts/run_notebooks.sh

# optional: limit execution to a subset of notebooks
FLOODSR_NOTEBOOK_IMAGE="${IMAGE_NAME}" FLOODSR_NOTEBOOK_PATTERN="tutorial_*.ipynb" bash docs/user/scripts/run_notebooks.sh
```
