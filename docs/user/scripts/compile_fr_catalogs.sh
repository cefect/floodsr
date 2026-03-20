#!/usr/bin/env bash

# Compile the tracked fr_CA translation catalogs to binary .mo files for Sphinx.
set -euo pipefail

# Resolve the docs/user directory relative to this script location.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docs_dir="$(cd "${script_dir}/.." && pwd)"
catalog_root="${docs_dir}/locale/fr_CA/LC_MESSAGES"
count=0

printf "Compiling fr_CA catalogs for Sphinx\n"
printf "docs_dir: %s\n" "${docs_dir}"
printf "catalog_root: %s\n" "${catalog_root}"

# Compile each French catalog in place so the translated HTML build can load it.
while IFS= read -r fp; do
  mo_fp="${fp%.po}.mo"
  count=$((count + 1))
  printf "[%02d] %s\n" "${count}" "${fp}"
  printf "     -> %s\n" "${mo_fp}"
  msgfmt "${fp}" -o "${mo_fp}"
done < <(find "${catalog_root}" -name "*.po" | sort)

printf "Compiled %d fr_CA catalog(s).\n" "${count}"
