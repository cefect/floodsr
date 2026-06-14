#!/usr/bin/env bash

# Copy only French .po catalogs with llm_draft entries for human review.
set -euo pipefail

# Resolve docs/user paths relative to this script.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docs_dir="$(cd "${script_dir}/.." && pwd)"

python - "${script_dir}" "${docs_dir}" <<'PY'
import sys
from pathlib import Path

from babel.messages import pofile

script_dir = Path(sys.argv[1])
docs_dir = Path(sys.argv[2])
sys.path.insert(0, str(script_dir))

from sync_fr_translations import copy_review_catalogs

catalog_root = docs_dir / "locale" / "fr" / "LC_MESSAGES"
review_dir = docs_dir / "_fr_review"
report_row_l = []

# Scan current catalogs and flag files that still need human review.
for po_fp in sorted(catalog_root.rglob("*.po")):
    with po_fp.open(encoding="utf-8") as stream:
        catalog = pofile.read_po(stream, locale="fr")
    for message in catalog:
        metadata = {}
        for comment in message.auto_comments:
            if ":" not in comment:
                continue
            key, value = comment.split(":", 1)
            metadata[key.strip()] = value.strip()
        if metadata.get("review_status") == "llm_draft":
            report_row_l.append({"catalog": po_fp.relative_to(catalog_root).as_posix(), "status": "llm_draft"})
            break

stats = copy_review_catalogs(catalog_root=catalog_root, review_dir=review_dir, report_row_l=report_row_l)
print(f"Copied {stats['review_catalogs']} catalog(s) to {stats['review_dir']}")
PY
