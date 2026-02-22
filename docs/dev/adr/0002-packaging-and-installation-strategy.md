# ADR-0002: Packaging and Installation Strategy

 
End users need a clean install path that does not pollute host environments (including QGIS-managed Python), with a CPU-only runtime.

 

## background
There is one real-world problem you should design around:

1. **Multiple ORT versions on the system can cause undefined behavior**

* ORT’s build docs explicitly warn that if multiple ORT versions are installed and library search paths are involved, ORT can find the wrong libraries and behave unpredictably.  

Because of that, standardize on one runtime package and avoid mixed ORT installs.

## decision
 
- Ship one pip package providing:
  - Python library: `floodsr`
  - CLI entrypoint: `floodsr`
- Recommend `pipx` as the primary installation method for end users.
- Keep publishing target as PyPI/TestPyPI wheels and source distributions.
- Validate published artifacts with an isolated smoke test that uses `pipx` (local or containerized).

## deployment strategy

1. Build and validate artifacts from source.
2. Upload to TestPyPI first.
3. Smoke test install with `pipx` from TestPyPI in an isolated runtime.
4. Promote the same process to PyPI after TestPyPI verification.

Reference install commands:

```bash
# install from TestPyPI into an isolated pipx venv
pipx install --index-url https://test.pypi.org/simple/ --pip-args="--extra-index-url https://pypi.org/simple" floodsr

# sanity checks
pipx runpip floodsr show floodsr
pipx run floodsr doctor
pipx run floodsr models list

# clean up local smoke-test environment
pipx uninstall floodsr
```


## Consequences

- Users get isolated installs by default.
- Runtime selection is explicit and CPU-only.
- This reduces the risk of ORT library conflicts from mixed installs.  
- Release validation more closely matches end-user CLI installation behavior.
