# USER documentation

- [ReadTheDocs](https://app.readthedocs.org/projects/floodsr/)

## Read the Docs config

- `.readthedocs.yaml`

## local build with sphinx

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

 
 