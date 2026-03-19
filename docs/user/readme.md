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
NOTE: needs to be done from main .devcontainer (not docs)
```bash
# run from the repository root in the notebook-capable dev environment
cd /workspace

# quick-start tutorial; runs from cache and copies the executed notebook back
conda run -n dev bash docs/user/notebooks/tutorial_1.sh

# large-raster tutorial; also runs from cache and copies the executed notebook back
conda run -n dev bash docs/user/notebooks/tutorial_2.sh

 
```
