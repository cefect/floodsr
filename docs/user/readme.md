# USER documentation

- [ReadTheDocs](https://app.readthedocs.org/projects/floodsr/)
- [ReadTheDocs french project](https://app.readthedocs.org/projects/floodsr-fr/)

## Read the Docs config

- `.readthedocs.yaml`
- French translations now live under `docs/user/locale/fr/LC_MESSAGES/` so RTD and local Sphinx builds use the same plain `fr` language slug.

# LOCAL COMPILE with SPHINX

## main (english)

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

 ## french

```bash
# 1) go to the docs source directory
cd /workspace/docs/user

# compile the fr .po catalogs to .mo files
bash scripts/compile_fr_catalogs.sh
 
# 3) build the fr html docs into a separate build directory
python -m sphinx -E -b html -D language=fr . "_build/fr_html"

# launch the fr index.html in the default Windows browser (from WSL)
\\wsl.localhost\Ubuntu\home\cefect\LS\09_REPOS\04_TOOLS\floodsr\docs\user\_build\fr_html\index.html
```




# MAINTAIN

## TRANSLATION REFRESH

Follow [ADR-0018: Docs and Tutorials Strategy](../dev/adr/0018-docs-and-tutorials.md) for the translation policy and review expectations.

```bash
# 1) go to the docs source directory
cd /workspace/docs/user

# 2) manually review/edit the existing fr catalogs


# 3) compile the edited .po catalogs to .mo files
bash scripts/compile_fr_catalogs.sh

 
```



## updating `docs/user/cli_reference.rst`

This page is maintained manually for docs builds.
If the CLI changes and you want to refresh the page from the live parser metadata, run:

```bash
cd /workspace
python docs/user/scripts/build_cli_reference.py
```

## update tutorial notebooks
NOTE: needs to be done from main .devcontainer (not docs)

Each tutorial runner stages execution in a temp-backed sandbox, keeps notebook side files there, and then copies the completed `.ipynb` back into `docs/user/notebooks`. 
Tutorial 3 is the exception for heavy reuse: it still points HRDEM/model caching at the project cache while keeping the notebook run directory in temp.

using pruned/curated plot-only outputs now

```bash
# run from the repository root in the notebook-capable dev environment
cd /workspace

# quick-start tutorial; runs from temp and copies the executed notebook back
conda run -n dev bash docs/user/notebooks/tutorial_1.sh

# plotting and CLI-options tutorial; runs from temp and copies the executed notebook back
conda run -n dev bash docs/user/notebooks/tutorial_2.sh

# large-raster tutorial; runs from temp, but keeps heavy HRDEM/model cache reuse
conda run -n dev bash docs/user/notebooks/tutorial_3.sh

 
```
