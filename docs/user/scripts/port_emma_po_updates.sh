#!/usr/bin/env bash

# Port Emma's translated fr catalogs into the tracked docs catalogs, then compile.
set -euo pipefail

# Resolve the docs/user paths relative to this script.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docs_dir="$(cd "${script_dir}/.." && pwd)"
target_dir="${docs_dir}/locale/fr/LC_MESSAGES"
compile_script="${script_dir}/compile_fr_catalogs.sh"
source_dir="${1:-/home/cefect/LS/10_IO/floodsr/docs/2026 03 29 - Emma - po updates}"

printf "Porting Emma fr catalogs into tracked docs catalogs\n"
printf "source_dir: %s\n" "${source_dir}"
printf "target_dir: %s\n" "${target_dir}"

# Fail early when the expected source or target tree is missing.
test -d "${source_dir}"
test -d "${target_dir}"
test -x "${compile_script}"

# Copy msgstr values by msgid while keeping the repo headers and file structure.
python - "${source_dir}" "${target_dir}" <<'PY'
import ast
import json
import re
import sys
from pathlib import Path


def split_blocks(text):
    """Split a PO file into entry blocks."""
    block_l = []
    lines = []
    for line in text.splitlines():
        if line.strip():
            lines.append(line)
            continue
        if lines:
            block_l.append(lines)
            lines = []
    if lines:
        block_l.append(lines)
    return block_l


def parse_block(lines):
    """Parse one PO block into a small editable payload."""
    data = {"msgctxt": None, "msgid": None, "msgstr": {}, "msgstr_order": [], "msgstr_index": None}
    field = None
    field_index = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("msgctxt"):
            field = "msgctxt"
            field_index = i
            data["msgctxt"] = ast.literal_eval(stripped[len("msgctxt"):].strip())
            continue
        if stripped.startswith("msgid_plural"):
            field = "msgid_plural"
            field_index = i
            continue
        if stripped.startswith("msgid"):
            field = "msgid"
            field_index = i
            data["msgid"] = ast.literal_eval(stripped[len("msgid"):].strip())
            continue
        if stripped.startswith("msgstr["):
            field_index = i if data["msgstr_index"] is None else data["msgstr_index"]
            data["msgstr_index"] = field_index
            match = re.match(r"msgstr\[(\d+)\]\s+(.*)", stripped)
            index = int(match.group(1))
            data["msgstr"][index] = ast.literal_eval(match.group(2))
            if index not in data["msgstr_order"]:
                data["msgstr_order"].append(index)
            field = f"msgstr[{index}]"
            continue
        if stripped.startswith("msgstr"):
            field = "msgstr"
            field_index = i
            data["msgstr_index"] = i
            data["msgstr"][0] = ast.literal_eval(stripped[len("msgstr"):].strip())
            if 0 not in data["msgstr_order"]:
                data["msgstr_order"].append(0)
            continue
        if stripped.startswith('"'):
            value = ast.literal_eval(stripped)
            if field == "msgctxt":
                data["msgctxt"] = (data["msgctxt"] or "") + value
            elif field == "msgid":
                data["msgid"] = (data["msgid"] or "") + value
            elif field == "msgstr":
                data["msgstr"][0] += value
            elif field and field.startswith("msgstr["):
                index = int(field[7:-1])
                data["msgstr"][index] += value
            continue
    return data


def normalize_msgstr(value):
    """Apply glossary fixes so imported text respects the docs ADR and local usage."""
    value = re.sub(r"\bMNEHR\b", "HRDEM", value)
    value = re.sub(r"\bMNE\b", "DEM", value)
    return value


def render_msgstr(data):
    """Render msgstr lines back into PO syntax."""
    index_l = data["msgstr_order"] or sorted(data["msgstr"])
    rendered = []
    for index in index_l:
        value = normalize_msgstr(data["msgstr"].get(index, ""))
        prefix = "msgstr" if index == 0 and index_l == [0] else f"msgstr[{index}]"
        rendered.append(f"{prefix} {json.dumps(value, ensure_ascii=False)}")
    return rendered


def block_key(data):
    """Build a stable lookup key for one entry."""
    return (data["msgctxt"], data["msgid"])


source_dir = Path(sys.argv[1])
target_dir = Path(sys.argv[2])
stats = {"files": 0, "entries_changed": 0, "glossary_hits": 0, "missing_files": 0, "missing_entries": 0}

for source_fp in sorted(source_dir.glob("*.po")):
    target_fp = target_dir / source_fp.name
    if not target_fp.exists():
        print(f"WARNING: no local catalog file for {source_fp.name}")
        stats["missing_files"] += 1
        continue
    source_block_l = split_blocks(source_fp.read_text(encoding="utf-8"))
    source_map = {}
    for block in source_block_l:
        data = parse_block(block)
        if data["msgid"]:
            source_map[block_key(data)] = data

    target_block_l = split_blocks(target_fp.read_text(encoding="utf-8"))
    target_key_s = set()
    for block in target_block_l:
        data = parse_block(block)
        if data["msgid"]:
            target_key_s.add(block_key(data))
    out_block_l = []
    file_changes = 0
    for block in target_block_l:
        data = parse_block(block)
        if not data["msgid"] or data["msgstr_index"] is None:
            out_block_l.append(block)
            continue
        source_data = source_map.get(block_key(data))
        if source_data is None:
            out_block_l.append(block)
            continue
        rendered = render_msgstr(source_data)
        glossary_before = "\n".join(source_data["msgstr"].values())
        stats["glossary_hits"] += glossary_before.count("MNEHR") + len(re.findall(r"\bMNE\b", glossary_before))
        new_block = block[: data["msgstr_index"]] + rendered
        if new_block != block:
            file_changes += 1
        out_block_l.append(new_block)

    for key, source_data in source_map.items():
        if key not in target_key_s:
            print(f"WARNING: no local entry for {target_fp.name}: {source_data['msgid'][:120]!r}")
            stats["missing_entries"] += 1

    target_fp.write_text("\n\n".join("\n".join(lines) for lines in out_block_l) + "\n", encoding="utf-8")
    stats["files"] += 1
    stats["entries_changed"] += file_changes
    print(f"{target_fp.name}: updated {file_changes} entry(ies)")

print(json.dumps(stats, ensure_ascii=False))
PY

# Compile the updated catalogs into .mo build artifacts.
"${compile_script}"
