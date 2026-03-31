# Tutorial Notebooks

This folder holds the committed tutorial notebook artifacts rendered into the user docs.

- For the governing docs/tutorial decisions, see [ADR-0018](../../dev/adr/0018-docs-and-tutorials.md).
- For the main user-facing entry points, see [tutorials.rst](../tutorials.rst) and [getting_started.rst](../getting_started.rst).

## sphinx rendering

Sphinx renders these `.ipynb` files directly with `myst_nb`; the notebook sources under `docs/user/notebooks/` are the documentation inputs, not generated markdown stand-ins.

Notebook execution is disabled during docs builds, so Sphinx renders the committed notebook state rather than running cells at build time. 
In practice, that means the committed notebooks should already be refreshed and pruned before a docs build: keep plot/image outputs that help the docs, drop noisy textual outputs when possible, and use tags such as `remove-input` for short validation-only cells that should execute during proofing but not appear in the rendered docs.

## environments

To re-run or refresh these tutorial notebooks from the repo, use the `main` devcontainer rather than the `docs` devcontainer. 
The notebook runners assume a notebook-capable environment with `jupyter` available, and in this repo that means the `dev` conda environment layered on top of the main runtime image. 


The notebooks themselves present the following user-facing execution contexts:

| Tutorial | CLI (`pipx`) | Local notebook (Jupyter) | Hosted notebook (Colab) | Repo proofing (`main` devcontainer + `conda -n dev`) |
| --- | --- | --- | --- | --- |
| Tutorial 1 | Yes | Yes | Yes | Yes |
| Tutorial 2 | Yes | Yes | Yes | Yes |
| Tutorial 3 | Yes, via extended install | Yes, via extended install | Yes, but marked experimental / not recommended | Yes |

## re-running the tutorials
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

# or do them all at once
conda run -n dev bash docs/user/notebooks/tutorial_1.sh && \
conda run -n dev bash docs/user/notebooks/tutorial_2.sh && \
conda run -n dev bash docs/user/notebooks/tutorial_3.sh

 
```
