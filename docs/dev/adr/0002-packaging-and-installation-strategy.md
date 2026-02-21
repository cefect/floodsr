# ADR-0002: Packaging and Installation Strategy

 
End users need a clean install path that does not pollute host environments (including QGIS-managed Python), with a CPU-only runtime.

 

## background
There is one real-world problem you should design around:

1. **Multiple ORT versions on the system can cause undefined behavior**

* ORT’s build docs explicitly warn that if multiple ORT versions are installed and library search paths are involved, ORT can find the wrong libraries and behave unpredictably.  

Because of that, standardize on one runtime package and avoid mixed ORT installs.

## decision
Use a “core + thin installer package” approach:

* `floodsr` (core code, **no hard dependency** on ORT)
* `floodsr-cpu` depends on:

  * `floodsr`
  * `onnxruntime` ([ONNX Runtime][2])
That way users install one supported runtime package:

```bash
pip install floodsr-cpu
```

- Ship one pip package providing:
  - Python library: `floodsr`
  - CLI entrypoint: `floodsr`
- Recommend `pipx` as the primary installation method for end users.


## Consequences

- Users get isolated installs by default.
- Runtime selection is explicit and CPU-only.
- This reduces the risk of ORT library conflicts from mixed installs.  

