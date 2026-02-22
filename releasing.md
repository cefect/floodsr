# releasing/publishing

## manual publish to test pypi

### setting up tokens
create a test token
copy to `~/.pypirc`:

```ini
[testpypi]
  username = __token__
  password = ....
```
ensure its bind mounted and accessible in the container



### Build, Test, and Publish (TestPyPI)

```bash
# 1) activate the project runtime environment (base per project guidance)
conda activate dev

# 2) check local package is installed in the active environment
python -c "from importlib.metadata import version; print(version('floodsr'))"

# 3) check build/publish tools are installed
python -m pip show build twine

# 4) run tests before packaging
pytest -q

# 5) build source + wheel into ./dist
python -m build

# 6) validate package metadata/artifacts
python -m twine check dist/*

# 7) publish to TestPyPI (requires ~/.pypirc with [testpypi] token)
python -m twine upload --repository testpypi dist/*

```

 
### optional: disposable container smoke test (no host pollution)

```bash
docker run --rm condaforge/miniforge3:25.3.1-0 bash -lc "
  set -euo pipefail &&
  export PIPX_HOME=/opt/pipx &&
  export PIPX_BIN_DIR=/usr/local/bin &&
  python -m pip install --upgrade pip pipx &&
  pipx install --index-url https://test.pypi.org/simple/ --pip-args='--extra-index-url https://pypi.org/simple' floodsr &&
  pipx runpip floodsr show floodsr &&
  floodsr doctor &&
  floodsr models list
"
```
