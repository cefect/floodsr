# ADR-0002: Packaging and Installation Strategy

 
End users need a clean install path that does not pollute host environments (including QGIS-managed Python), while allowing CPU/GPU runtime variants.

 

## background
There are two real-world problems you should design around:

1. **Resolvers may install both CPU and GPU distributions**

* ORT has an open issue noting `onnxruntime-gpu` does not declare that it “provides” the `onnxruntime` distribution in wheel metadata, so dependency resolvers can treat them as unrelated and install both.  

2. **Multiple ORT versions on the system can cause undefined behavior**

* ORT’s build docs explicitly warn that if multiple ORT versions are installed and library search paths are involved, ORT can find the wrong libraries and behave unpredictably.  

Because of that, avoid a design where your project *requires* `onnxruntime` while also trying to support `onnxruntime-gpu` as a drop-in alternative.

## decision
Use a “core + thin variant packages” approach:

* `floodsr` (core code, **no hard dependency** on ORT)
* `floodsr-cpu` depends on:

  * `floodsr`
  * `onnxruntime` ([ONNX Runtime][2])
* `floodsr-cuda` depends on:

  * `floodsr`
  * `onnxruntime-gpu` ([ONNX Runtime][2])

That way users choose exactly one:

```bash
pip install floodsr-cpu
# or
pip install floodsr-cuda
```

- Ship one pip package providing:
  - Python library: `floodsr`
  - CLI entrypoint: `floodsr`
- Recommend `pipx` as the primary installation method for end users.


## Consequences

- Users get isolated installs by default.
- Runtime variants are explicit and conflict-aware (`onnxruntime` vs `onnxruntime-gpu`).
- Packaging remains flexible as GPU support expands.
- This avoids the “both installed” situation that the ORT packaging issue can trigger.  



