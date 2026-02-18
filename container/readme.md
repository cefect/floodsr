# development containers

## probe the upstream
```bash
upstream=mcr.microsoft.com/devcontainers/python:3.12-bookworm

# fetch the upstream image
docker pull "$upstream"

 

docker run --rm --entrypoint /bin/bash "$upstream" -lc "python -m pip freeze" \
  > container/pip-freeze.upstream.txt

```
## build base
```bash
export IMAGE_NAME='cefect/floodsr:base-v0.1'

docker buildx build --load -t $IMAGE_NAME -f container/Dockerfile --target base .

```

export the pip freeze
```bash
docker run --rm --entrypoint /bin/bash "$IMAGE_NAME" -lc "python -m pip freeze" \
  > container/pip-freeze.base.txt
```


## build dev
```bash
export IMAGE_NAME='cefect/floodsr:dev-v0.2'

docker buildx build --load -t $IMAGE_NAME -f container/Dockerfile --target dev .

```

export the pip freeze
```bash
docker run --rm --entrypoint /bin/bash "$IMAGE_NAME" -lc "python -m pip freeze" \
  > container/pip-freeze.dev.txt
```


update the .devcontainer/compose
```bash
 
yq -y -i '.services.dev.image = env.IMAGE_NAME' .devcontainer/docker-compose.yml
```
