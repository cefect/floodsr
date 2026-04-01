#!/usr/bin/env bash

# Run the Sphinx docs build from inside the docs container.
set -euo pipefail

docs_dir="/workspace/docs/user"
build_dir="_build/manual"
sphinx_args=(-b html . "${build_dir}")

# Switch to the translated docs pipeline when requested by the host wrapper.
if [[ "${1:-}" == "--french" ]]; then
  bash scripts/compile_fr_catalogs.sh
  build_dir="_build/fr_html"
  sphinx_args=(-E -b html -D language=fr . "${build_dir}")
fi

cd "${docs_dir}"

# Show the active Sphinx runtime before building so the container state is explicit.
python -m sphinx --version
python -m sphinx "${sphinx_args[@]}"

# Echo the built landing page so the host caller can open the result directly.
printf "index.html: %s\n" "${docs_dir}/${build_dir}/index.html"
