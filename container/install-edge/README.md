# install-edge containers

Build and push commands for the install-proof notebook context images.

## tags

```bash
tag="v0.1"
export JUPYTER_IMAGE="cefect/floodsr:install-edge-jupyter-${tag}"
export COLAB_IMAGE="cefect/floodsr:install-edge-colab-${tag}"
```

## build

```bash
docker buildx build --platform linux/amd64 --load -t "${JUPYTER_IMAGE}" -f container/install-edge/Dockerfile.jupyter .
docker buildx build --platform linux/amd64 --load -t "${COLAB_IMAGE}" -f container/install-edge/Dockerfile.colab .
```

## push

```bash
docker buildx build --platform linux/amd64 --push -t "${JUPYTER_IMAGE}" -f container/install-edge/Dockerfile.jupyter .
docker buildx build --platform linux/amd64 --push -t "${COLAB_IMAGE}" -f container/install-edge/Dockerfile.colab .
```
