# development containers

## python interpreter notes

In this devcontainer, seeing multiple `which -a python python3` hits is expected.
The same path can appear multiple times when `PATH` has duplicate entries, and Debian also keeps a system Python under `/usr/bin` for OS tooling.

Typical result:
- user/default Python: `/usr/local/bin/python3.12` (what `python` and `python3` resolve to in this container)
- system Python: `/usr/bin/python3.11` (used by distro-level scripts/packages)

Quick checks:
```bash
which -a python python3
which -a python python3 | xargs -I{} readlink -f {} | sort -u
printf '%s\n' "$PATH" | tr ':' '\n' | nl -ba
```

Gotchas/warnings:
- do not change `/usr/bin/python3` symlink; Debian tools may break
- avoid mixing `pip` from one interpreter with `python` from another
- prefer `python -m pip ...` over bare `pip ...`
- for scripts/jobs, pin the interpreter explicitly when needed (example: `/usr/local/bin/python3.12`)
- if package behavior looks inconsistent, confirm both interpreter and pip paths first:
```bash
python -c "import sys; print(sys.executable)"
python -m pip --version
```

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
export IMAGE_NAME='cefect/floodsr:dev-v0.3'

docker buildx build --load -t $IMAGE_NAME -f container/Dockerfile --target dev .

```

export the pip freeze
```bash
docker run --rm --entrypoint /bin/bash "$IMAGE_NAME" -lc "python -m pip freeze" \
  > container/pip-freeze.dev.txt
```


update the main devcontainer compose
```bash
 
yq -y -i '.services.dev.image = env.IMAGE_NAME' .devcontainer/main/docker-compose.yml
```
