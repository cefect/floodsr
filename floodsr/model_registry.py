"""Model manifest resolution, worker discovery, and retrieval backends."""

import importlib.util, json, logging, os, shutil, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from floodsr.cache_paths import get_model_cache_path
from floodsr.checksums import assert_sha256, verify_sha256


DEFAULT_MANIFEST_FP = Path(__file__).with_name("models.json")
log = logging.getLogger(__name__)


def _stream_response_to_destination(response, destination: Path, logger=None, chunk_size: int = 1024 * 1024) -> Path:
    """Stream an HTTP response to disk with a simple progress bar when possible."""
    log = logger or logging.getLogger(__name__)

    # Parse response size so progress can report percent complete.
    total_bytes = response.headers.get("Content-Length")
    try:
        total_size = int(total_bytes) if total_bytes else None
    except ValueError:
        total_size = None

    # Only draw a progress bar for TTY stderr with known total size.
    show_progress = bool(total_size) and sys.stderr.isatty()
    downloaded = 0
    with destination.open("wb") as stream:
        chunk = response.read(chunk_size)
        while chunk:
            stream.write(chunk)
            downloaded += len(chunk)

            if show_progress and total_size:
                width = 30
                ratio = min(downloaded / total_size, 1.0)
                filled = int(width * ratio)
                bar = "#" * filled + "-" * (width - filled)
                sys.stderr.write(
                    f"\r[{bar}] {ratio * 100:6.2f}% ({downloaded:,}/{total_size:,} bytes)"
                )
                sys.stderr.flush()

            chunk = response.read(chunk_size)

    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()
    log.debug(f"downloaded {downloaded:,} bytes to\n    {destination}")
    return destination


def get_github_auth_token(logger=None) -> str | None:
    """Resolve a GitHub token from env vars first, then gh CLI auth state."""
    log = logger or logging.getLogger(__name__)

    # Prefer explicit environment variables for reproducible non-interactive runs.
    for env_var in ("FLOODSR_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        token = os.environ.get(env_var)
        if token:
            log.debug(f"using GitHub token from ${env_var}")
            return token

    # Fall back to GitHub CLI credentials when users are already logged in via `gh auth login`.
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        log.debug("gh CLI not available; no GitHub token discovered")
        return None
    except subprocess.CalledProcessError as err:
        log.debug(f"gh auth token failed with exit code {err.returncode}; no GitHub token discovered")
        return None

    token = result.stdout.strip()
    if token:
        log.debug("using GitHub token from gh auth token")
        return token
    log.debug("gh auth token returned empty output")
    return None


@dataclass(frozen=True)
class ModelRecord:
    """Resolved model metadata from the weights manifest."""

    version: str
    file_name: str
    url: str
    sha256: str
    description: str = ""


class WeightsRetrievalBackend:
    """Abstract retrieval backend for fetching model bytes."""

    name = "base"

    def retrieve(self, source: str, destination: Path) -> Path:
        """Fetch model bytes from source into destination."""
        raise NotImplementedError


class HttpRetrievalBackend(WeightsRetrievalBackend):
    """Retrieve model weights from HTTP(S) sources."""

    name = "http"

    def retrieve(self, source: str, destination: Path) -> Path:
        """Download bytes from an HTTP(S) URL to destination."""
        assert source, "source cannot be empty"
        assert isinstance(destination, Path), "destination must be a pathlib.Path"

        parsed = urlparse(source)
        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError(f"unsupported scheme for http backend: {parsed.scheme}")

        # Parse GitHub release-style URLs once for specialized fallback behavior/messages.
        path_parts = [part for part in parsed.path.split("/") if part]
        is_github_release_url = (
            parsed.netloc.lower() == "github.com"
            and len(path_parts) >= 6
            and path_parts[2] == "releases"
            and path_parts[3] == "download"
        )

        # Stream response bodies directly to disk to avoid loading whole models into memory.
        destination.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"attempting unauthenticated model download from\n    {source}")

        # First attempt: no credentials (works for public assets and avoids needless token use).
        try:
            with urlopen(Request(source)) as response:  # nosec B310
                _stream_response_to_destination(response, destination, logger=log)
            return destination
        except HTTPError as err:
            unauthenticated_http_error = err
            log.info(f"unauthenticated download failed with HTTP {err.code}; attempting credentialed fallback")
        except URLError as err:
            raise RuntimeError(f"failed to download model from '{source}' ({err})") from err

        # Second attempt: discover credentials (env first, then `gh auth token`) and retry.
        auth_token = get_github_auth_token(logger=log)
        if not auth_token:
            message = f"failed to download model from '{source}' (HTTP {unauthenticated_http_error.code})"
            if is_github_release_url:
                message = (
                    f"{message}. If this is a private GitHub release asset, run "
                    f"'gh auth login' or set FLOODSR_GITHUB_TOKEN/GITHUB_TOKEN."
                )
            raise RuntimeError(message) from unauthenticated_http_error

        log.info(f"retrying model download with token auth from\n    {source}")
        request_headers = {"Authorization": f"Bearer {auth_token}"}
        request = Request(source, headers=request_headers)
        try:
            with urlopen(request) as response:  # nosec B310
                _stream_response_to_destination(response, destination, logger=log)
            return destination
        except HTTPError as err:
            # Private GitHub release assets can still 404 on the web URL, so resolve via API.
            if err.code == 404 and is_github_release_url:
                owner, repo, _, _, tag = path_parts[:5]
                asset_name = "/".join(path_parts[5:])
                release_api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
                log.debug(f"retrying via GitHub release API for {owner}/{repo} tag '{tag}'")
                release_request = Request(
                    release_api_url,
                    headers={
                        "Accept": "application/vnd.github+json",
                        "Authorization": f"Bearer {auth_token}",
                    },
                )
                with urlopen(release_request) as release_response:  # nosec B310
                    release_payload = json.loads(release_response.read().decode("utf-8"))
                asset_url = next(
                    (
                        asset["url"]
                        for asset in release_payload.get("assets", [])
                        if asset.get("name") == asset_name
                    ),
                    None,
                )
                if not asset_url:
                    raise RuntimeError(f"release asset '{asset_name}' not found for tag '{tag}' ({source})") from err
                asset_request = Request(
                    asset_url,
                    headers={
                        "Accept": "application/octet-stream",
                        "Authorization": f"Bearer {auth_token}",
                    },
                )
                with urlopen(asset_request) as asset_response:  # nosec B310
                    _stream_response_to_destination(asset_response, destination, logger=log)
                return destination

            message = f"failed to download model from '{source}' (HTTP {err.code})"
            if is_github_release_url:
                message = (
                    f"{message}. If this is a private GitHub release asset, set "
                    f"FLOODSR_GITHUB_TOKEN or GITHUB_TOKEN."
                )
            raise RuntimeError(message) from err
        except URLError as err:
            raise RuntimeError(f"failed to download model from '{source}' ({err})") from err


class FileRetrievalBackend(WeightsRetrievalBackend):
    """Retrieve model weights from file paths or file:// URIs."""

    name = "file"

    def retrieve(self, source: str, destination: Path) -> Path:
        """Copy model bytes from a local path into destination."""
        parsed = urlparse(source)
        if parsed.scheme.lower() in {"", "file"}:
            source_fp = (
                Path(f"//{parsed.netloc}{unquote(parsed.path)}")
                if parsed.netloc
                else Path(unquote(parsed.path) or source)
            )
        else:
            raise ValueError(f"unsupported scheme for file backend: {parsed.scheme}")
        source_fp = source_fp.expanduser().resolve()
        if not source_fp.exists():
            raise FileNotFoundError(f"source model not found: {source_fp}")

        # Copy bytes to a destination managed by the caller.
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_fp, destination)
        return destination


def load_models_manifest(manifest_fp: str | Path | None = None) -> dict:
    """Load the model manifest from disk."""
    manifest_path = Path(manifest_fp).expanduser().resolve() if manifest_fp else DEFAULT_MANIFEST_FP
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")

    # Read JSON manifest and return the payload.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("manifest field 'models' must be a dictionary")
    return models


def list_models(manifest_fp: str | Path | None = None) -> list[ModelRecord]:
    """Return all models defined in the manifest."""
    records = []
    for version, payload in sorted(load_models_manifest(manifest_fp).items()):
        records.append(
            ModelRecord(
                version=version,
                file_name=payload["file_name"],
                url=payload["url"],
                sha256=payload["sha256"],
                description=payload.get("description", ""),
            )
        )
    return records


def resolve_model(model_version: str, manifest_fp: str | Path | None = None) -> ModelRecord:
    """Resolve one model entry from the manifest."""
    assert model_version, "model_version cannot be empty"
    models = load_models_manifest(manifest_fp)
    if model_version not in models:
        available = ", ".join(sorted(models))
        raise KeyError(f"model '{model_version}' not found. available: {available}")

    # Normalize the selected model payload into a typed record.
    payload = models[model_version]
    return ModelRecord(
        version=model_version,
        file_name=payload["file_name"],
        url=payload["url"],
        sha256=payload["sha256"],
        description=payload.get("description", ""),
    )


def get_retrieval_backend(source_url: str, backend_name: str | None = None) -> WeightsRetrievalBackend:
    """Select a retrieval backend from explicit name or URL scheme."""
    if backend_name == "http":
        return HttpRetrievalBackend()
    if backend_name == "file":
        return FileRetrievalBackend()
    if backend_name is not None:
        raise ValueError(f"unsupported backend '{backend_name}'")

    # Derive backend selection from URI scheme when no override is provided.
    scheme = urlparse(source_url).scheme.lower()
    if scheme in {"http", "https"}:
        return HttpRetrievalBackend()
    if scheme in {"", "file"}:
        return FileRetrievalBackend()
    raise ValueError(f"unable to select backend for URL scheme '{scheme}'")


def fetch_model(
    model_version: str,
    cache_dir: str | Path | None = None,
    manifest_fp: str | Path | None = None,
    backend_name: str | None = None,
    force: bool = False,
) -> Path:
    """Fetch one model to cache and verify its checksum."""
    model = resolve_model(model_version, manifest_fp=manifest_fp)
    model_fp = get_model_cache_path(model.version, model.file_name, cache_dir=cache_dir)
    part_fp = model_fp.with_suffix(f"{model_fp.suffix}.part")

    # Reuse an existing cached model only when checksum validation passes.
    if model_fp.exists() and not force and verify_sha256(model_fp, model.sha256):
        return model_fp

    # Download to a temporary file first and atomically replace on success.
    if part_fp.exists():
        part_fp.unlink()
    backend = get_retrieval_backend(model.url, backend_name=backend_name)
    try:
        backend.retrieve(model.url, part_fp)
        assert_sha256(part_fp, model.sha256)
        part_fp.replace(model_fp)
    finally:
        if part_fp.exists():
            part_fp.unlink()
    return model_fp


def get_model_worker_path(model_version: str) -> Path:
    """Return the expected worker module path for a model version."""
    assert model_version, "model_version cannot be empty"
    return Path(__file__).with_name("models") / f"{model_version}.py"


def model_worker_exists(model_version: str) -> bool:
    """Return whether a worker module file exists for this model version."""
    return get_model_worker_path(model_version).exists()


def list_runnable_model_versions(manifest_fp: str | Path | None = None) -> list[str]:
    """Return manifest model versions that have matching worker modules."""
    runnable_versions: list[str] = []
    for version in load_models_manifest(manifest_fp):
        if model_worker_exists(version):
            runnable_versions.append(version)
    return runnable_versions


def resolve_model_worker_class(model_version: str):
    """Load and return `ModelWorker` class for a model version."""
    worker_fp = get_model_worker_path(model_version)
    if not worker_fp.exists():
        raise FileNotFoundError(f"missing model worker module for '{model_version}': {worker_fp}")

    module_name = f"floodsr.models._worker_{model_version}"
    spec = importlib.util.spec_from_file_location(module_name, worker_fp)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load worker module spec from: {worker_fp}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker_class = getattr(module, "ModelWorker", None)
    if worker_class is None:
        raise AttributeError(f"worker module '{worker_fp}' must define `ModelWorker`")

    from floodsr.models.base import Model

    if not isinstance(worker_class, type) or not issubclass(worker_class, Model):
        raise TypeError(f"`ModelWorker` in '{worker_fp}' must subclass floodsr.models.base.Model")
    return worker_class
