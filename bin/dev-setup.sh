#!/usr/bin/env bash
# dev-setup.sh — One-shot developer setup for floodsr (Docker-based)
#
# What this does (derived from CONTRIBUTING.md):
#   1. Checks prerequisites (docker, git, git-lfs, gh)
#   2. Authenticates with GitHub and exports FLOODSR_GITHUB_TOKEN
#   3. Fetches Git LFS test data
#   4. Builds the dev image locally
#   5. Runs the lock-alignment test to confirm environment parity (amd64 only)
#
# Usage:
#   chmod +x dev-setup.sh && ./dev-setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

IMAGE_NAME="cefect/floodsr:miniforge-dev-v0.9"

# ── Colors ──────────────────────────────────────────────────────────
red()   { printf '\033[0;31m%s\033[0m\n' "$*"; }
green() { printf '\033[0;32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

# ── 1. Check prerequisites ─────────────────────────────────────────
bold "==> Checking prerequisites"

missing=()
for cmd in docker git gh; do
  if ! command -v "$cmd" &>/dev/null; then
    missing+=("$cmd")
  fi
done

if ! command -v git-lfs &>/dev/null && ! git lfs version &>/dev/null 2>&1; then
  missing+=("git-lfs")
fi

if [ ${#missing[@]} -gt 0 ]; then
  red "Missing required tools: ${missing[*]}"
  echo "Install them before re-running this script."
  echo "  brew install ${missing[*]}   # macOS"
  exit 1
fi

# Docker must be running
if ! docker info &>/dev/null; then
  red "Docker daemon is not running. Start Docker Desktop and retry."
  exit 1
fi

green "All prerequisites found."

# ── 2. GitHub auth + FLOODSR_GITHUB_TOKEN ──────────────────────────
bold "==> Setting up GitHub authentication"

if ! gh auth token &>/dev/null; then
  echo "You are not logged into the GitHub CLI. Launching 'gh auth login' …"
  gh auth login
fi

export FLOODSR_GITHUB_TOKEN
FLOODSR_GITHUB_TOKEN="$(gh auth token)"

if [ -z "$FLOODSR_GITHUB_TOKEN" ]; then
  red "Could not obtain a GitHub token from 'gh auth token'."
  exit 1
fi

green "GitHub token obtained."

# Persist into shell profile if not already there
SHELL_RC="$HOME/.zshrc"
[ -f "$HOME/.bashrc" ] && [ ! -f "$HOME/.zshrc" ] && SHELL_RC="$HOME/.bashrc"

if ! grep -q 'FLOODSR_GITHUB_TOKEN' "$SHELL_RC" 2>/dev/null; then
  echo ""
  echo "To make the token available in every new shell, add this to $SHELL_RC:"
  echo ""
  echo '  export FLOODSR_GITHUB_TOKEN="$(gh auth token)"'
  echo ""
  read -rp "Add it now? [y/N] " answer
  if [[ "$answer" =~ ^[Yy] ]]; then
    printf '\n# floodsr dev token\nexport FLOODSR_GITHUB_TOKEN="$(gh auth token)"\n' >> "$SHELL_RC"
    green "Added to $SHELL_RC"
  fi
fi

# ── 3. Git LFS test data ──────────────────────────────────────────
bold "==> Fetching Git LFS test data"

git lfs install --skip-smudge || true
git lfs pull --include="tests/data/**" --exclude=""

# Verify — no LFS pointer files should remain
pointers=$(grep -RIl --include="*.tif" --include="*.tiff" \
  "^version https://git-lfs.github.com/spec/v1$" tests/data 2>/dev/null || true)
if [ -n "$pointers" ]; then
  red "WARNING: LFS pointer files still present:"
  echo "$pointers"
  echo "Try: git lfs checkout tests/data"
else
  green "LFS data OK — no pointer files."
fi

# ── 4. Build the dev image locally ────────────────────────────────
bold "==> Building dev image: $IMAGE_NAME"

docker buildx build --load \
  --platform linux/amd64 \
  -f container/miniforge/Dockerfile \
  -t "$IMAGE_NAME" \
  --target dev \
  "$REPO_ROOT"

green "Image built successfully."

# ── 5. Run environment validation test ──────────────────────────
bold "==> Validating environment sanity (key packages importable)"

docker run --rm \
  --entrypoint /bin/bash \
  -v "$REPO_ROOT:/workspace:ro" \
  -w /workspace \
  "$IMAGE_NAME" \
  -lc 'conda run -n dev pytest -xvs tests/test_lock_alignment.py'

green "Environment validation passed!"

# ── 6. Summary & next steps ──────────────────────────────────────
bold "==> Setup complete!"
cat <<EOF

Image:  $IMAGE_NAME

Next steps:

  # Run an interactive dev shell:
  docker run --rm -it \\
    --entrypoint /bin/bash \\
    -v "$REPO_ROOT:/workspace" \\
    -e FLOODSR_GITHUB_TOKEN="\$FLOODSR_GITHUB_TOKEN" \\
    -w /workspace \\
    $IMAGE_NAME -l

  # Verify inside the container:
  python -m floodsr.cli models list
  python -m floodsr.cli models fetch ResUNet_16x_DEM
  pytest -q tests/test_model_registry.py

EOF
