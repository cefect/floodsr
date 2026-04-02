#!/usr/bin/env bash

# Compile the tracked fr translation catalogs to binary .mo files for Sphinx.
set -euo pipefail

# Resolve the docs/user directory relative to this script location.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docs_dir="$(cd "${script_dir}/.." && pwd)"
catalog_root="${docs_dir}/locale/fr/LC_MESSAGES"
count=0

# Prefer msgfmt when available, but allow a Babel-based fallback for local venv runs.
if command -v msgfmt >/dev/null 2>&1; then
  compile_cmd="msgfmt"
elif command -v pybabel >/dev/null 2>&1; then
  compile_cmd="pybabel"
else
  printf "Neither msgfmt nor pybabel is available on PATH.\n" >&2
  exit 127
fi

printf "Compiling fr catalogs for Sphinx\n"
printf "docs_dir: %s\n" "${docs_dir}"
printf "catalog_root: %s\n" "${catalog_root}"
printf "compiler: %s\n" "${compile_cmd}"

# Compile each French catalog in place so the translated HTML build can load it.
while IFS= read -r fp; do
  mo_fp="${fp%.po}.mo"
  count=$((count + 1))
  printf "[%02d] %s\n" "${count}" "${fp}"
  printf "     -> %s\n" "${mo_fp}"
  if [[ "${compile_cmd}" == "msgfmt" ]]; then
    msgfmt "${fp}" -o "${mo_fp}"
  else
    pybabel compile -l fr -i "${fp}" -o "${mo_fp}"
  fi
done < <(find "${catalog_root}" -name "*.po" | sort)

printf "Compiled %d fr catalog(s).\n" "${count}"
