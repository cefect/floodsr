# docs container

Build commands for the docs/Sphinx image.

## build base

```bash
export IMAGE_NAME='cefect/floodsr-docs:base-v0.1'
docker buildx build --load -t "$IMAGE_NAME" -f docs/container/Dockerfile --target base .
```

## build dev

```bash
export IMAGE_NAME='cefect/floodsr-docs:dev-v0.1'
docker buildx build --load -t "$IMAGE_NAME" -f docs/container/Dockerfile --target dev .
```

## quick docs build smoke check

```bash
docker run --rm -v "$PWD":/workspace -w /workspace --entrypoint /bin/bash "$IMAGE_NAME" \
  -lc "sphinx-build -b html docs docs/_build/html"
```
