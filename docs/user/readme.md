# documentation

## local build with sphinx

```bash
# 1) go to the docs source directory
cd /workspace/docs/user

# 2) verify sphinx is available in the current environment
python -m sphinx --version

# 3) build html docs from this directory into BUILD_DIR/html
export BUILD_DIR=/home/cefect/LS/10_IO/floodsr/docs
python -m sphinx -b html . "${BUILD_DIR}/html"

# 4) verify the main html output exists
ls -lh "${BUILD_DIR}/html/index.html"
```

## open built html in VS Code (quick check)

```bash
# 5) open the built html file directly in the current VS Code window
code "${BUILD_DIR}/html/readme.html"
```

## launch built html in a Windows web browser

### from WSL (outside container)

```bash
# 6) launch the html file in the default Windows browser
cmd.exe /C start "" "$(wslpath -w "${BUILD_DIR}/html/readme.html")"
```

### from the docs devcontainer (`.devcontainer/docs`)

```bash
# 7) serve docs on port 8000 (VS Code auto-forwards and opens Windows browser)
python -m http.server 8000 --directory "${BUILD_DIR}/html"
```

Then open `http://127.0.0.1:8000/readme.html` if it does not open automatically.
