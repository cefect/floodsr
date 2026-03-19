# install-edge containers

Build and push commands for the two install-proof context images.

## tags

```bash
tag="v0.2"
export MAIN_IMAGE="cefect/floodsr:install-edge-main-${tag}"
export COLAB_IMAGE="cefect/floodsr:install-edge-colab-${tag}"
```

## build

```bash
docker buildx build --platform linux/amd64 --load -t "${MAIN_IMAGE}" -f container/install-edge/Dockerfile.main .
docker buildx build --platform linux/amd64 --load -t "${COLAB_IMAGE}" -f container/install-edge/Dockerfile.colab .
```

## push

```bash
docker buildx build --platform linux/amd64 --push -t "${MAIN_IMAGE}" -f container/install-edge/Dockerfile.main .
docker buildx build --platform linux/amd64 --push -t "${COLAB_IMAGE}" -f container/install-edge/Dockerfile.colab .
```
