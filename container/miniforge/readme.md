# US Flood Events dev container


## BASE IMAGE=====================

```bash
#base image
docker run -it --rm condaforge/miniforge3:25.3.1-0 bash

 
 
```
 


## Build Images: deploy
from WSL:
```bash

#set the image name
IMAGE_NAME="cefect/floodsr:miniforge-deploy-v0.2"

# build the container
docker buildx build --load -f container/miniforge/Dockerfile -t "${IMAGE_NAME}" --target deploy .

```


explore w/ a random user
```bash
echo $IMAGE_NAME

 

# dump installed packages
docker run --rm -v "$PWD/container/miniforge:/out" "$IMAGE_NAME" \
  bash -lc "conda run -n deploy python -m pip freeze > /out/pip-freeze-deploy.txt && \
  conda env export -n deploy > /out/conda-env-deploy.lock.yml"

```

push to Docker Hub
```bash
# push
docker push $IMAGE_NAME

```

 
## Build Images: analysis
from WSL
```bash
IMAGE_NAME="cefect/floodsr:miniforge-dev-v0.2"
docker buildx build --load -f container/miniforge/Dockerfile -t "${IMAGE_NAME}" --target dev .
```

update the main devcontainer compose
```bash
 
yq -y -i '.services.dev.image = env.IMAGE_NAME' .devcontainer/main/docker-compose.yml
```
