# docs container

Build commands for the docs/Sphinx image.

 
## build dev

```bash
export IMAGE_NAME='cefect/floodsr-docs:dev-v0.2'
docker buildx build --load -t "$IMAGE_NAME" -f container/docs/Dockerfile --target dev .
```

dump installed packages
```bash
docker run --rm -v "$PWD/container/docs:/out" --entrypoint /bin/bash "$IMAGE_NAME" \
  -lc "python -m pip freeze --all > /out/pip-freeze-docs-dev.txt"

```
## quick docs build smoke check

```bash
docker run --rm -v "$PWD":/workspace -w /workspace --entrypoint /bin/bash "$IMAGE_NAME" \
  -lc "sphinx-build -b html docs/user docs/user/_build/html"
```


# .devcontainer
see `.devcontainer/docs`

update the devcontainer compose
```bash
 
yq -y -i '.services.dev.image = env.IMAGE_NAME' .devcontainer/docs/docker-compose.yml
```
