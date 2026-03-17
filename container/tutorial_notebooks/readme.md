# tutorial notebooks container

Build commands for the prebuilt tutorial notebook runner image.

## build

```bash
tag="v0.1"
export IMAGE_NAME="cefect/floodsr:tutorial-notebooks-${tag}"
docker buildx build --load -t "${IMAGE_NAME}" -f container/tutorial_notebooks/Dockerfile .
```

 
## notebook runner

```bash
export FLOODSR_NOTEBOOK_IMAGE="${IMAGE_NAME}"
bash docs/user/scripts/run_notebooks.sh
```
