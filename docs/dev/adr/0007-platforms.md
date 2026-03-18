# platform and provider support
CPU-first officially supported. GPU experimental but tested.

## platforms
In order of priority, we plan to support:
- unix
- windows
- macOS [deferred]

## required user specs
- python 3.12+ with pip installed
- pipx installed for local CLI-first installs
- google colab users should install with pip inside the notebook runtime
- enough memory to fit the output tile (but possibly not all the input tiles as well). >16GB?

## providers
In order of priority, we plan to support:
- ONNX Runtime (CPU)
- ONNX Runtime (CUDA) [deferred]
