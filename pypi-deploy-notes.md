# PyPI Deployment Runsheet — `floodsr`

## Prerequisites
- [ ] PyPI account created at https://pypi.org
- [ ] (Recommended) TestPyPI account at https://test.pypi.org
- [ ] API tokens generated for both (Account Settings → API tokens)
- [ ] Dev dependencies installed: `pip install -e ".[dev]"`

---

## 1. Configure PyPI credentials

Create or update `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-<your-token-here>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-<your-testpypi-token-here>
```

```bash
chmod 600 ~/.pypirc
```

---

## 2. Verify the codebase is ready

```bash
cd /Users/walter/floodsr

# Run tests
pytest

# Check code style
ruff check floodsr/
```

---

## 3. Tag the release (version comes from git tags via setuptools-scm)

```bash
# Check current version state
git log --oneline -5
git tag -l

# Create an annotated tag (e.g., v0.1.0)
git tag -a v0.1.0 -m "Release v0.1.0"

# Verify setuptools-scm resolves it correctly
python -m setuptools_scm
```

> Version format is driven by `tag_regex = "^v(?P<version>.+)$"` in `pyproject.toml`. Tag **must** start with `v`.

---

## 4. Build the distribution

```bash
# Clean any previous builds
rm -rf dist/ build/

# Build sdist + wheel
python -m build
```

Expected output in `dist/`:
```
floodsr-0.1.0.tar.gz
floodsr-0.1.0-py3-none-any.whl
```

---

## 5. Inspect the build (optional but recommended)

```bash
# Check wheel contents
python -m zipfile -l dist/floodsr-*.whl

# Check that models.json is included
python -m zipfile -l dist/floodsr-*.whl | grep models.json

# Validate with twine
twine check dist/*
```

---

## 6. Test upload to TestPyPI

```bash
twine upload --repository testpypi dist/*

# Verify install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ floodsr
```

---

## 7. Upload to PyPI (production)

```bash
twine upload dist/*
```

---

## 8. Verify the release

```bash
pip install floodsr
python -c "import floodsr; print(floodsr.__version__)"
```

---

## 9. Push the tag to remote

```bash
git push origin v0.1.0
```

---

## Notes

- **Version is automatic** — `setuptools-scm` derives it from the git tag. Don't set it manually.
- `models.json` is explicitly included via `package-data` in `pyproject.toml` — verify it's in the wheel (step 5).
- `twine check` will catch malformed metadata before upload.
- If you need to re-release the same version, delete the tag first: `git tag -d v0.1.0`
