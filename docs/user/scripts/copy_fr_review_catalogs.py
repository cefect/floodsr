"""Write French review-only PO catalogs for human review."""

import argparse, json, logging, sys
from pathlib import Path

from babel.messages import pofile

from sync_fr_translations import copy_review_catalogs


def main_copy_fr_review_catalogs(docs_dir: Path) -> dict:
    """Copy only catalogs with ``llm_draft`` entries into ``_fr_review``."""
    assert docs_dir.exists(), f"missing docs dir:\n    {docs_dir}"
    catalog_root = docs_dir / "locale" / "fr" / "LC_MESSAGES"
    review_dir = docs_dir / "_fr_review"
    assert catalog_root.exists(), f"missing catalog root:\n    {catalog_root}"

    log = logging.getLogger(__name__)
    log.info(f"scanning fr catalogs from\n    {catalog_root}")
    report_row_l = []
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
    log.info(f"wrote {stats['review_catalogs']} review catalog(s) to\n    {stats['review_dir']}")
    return stats


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the French review catalog copy."""
    parser = argparse.ArgumentParser(description="Write docs/user/_fr_review catalogs with only llm_draft entries.")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Sphinx user docs directory containing locale/fr/LC_MESSAGES.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed_args = _parse_arguments()
    logging.basicConfig(level=logging.DEBUG if parsed_args.debug else logging.INFO, format="%(message)s")
    summary = main_copy_fr_review_catalogs(docs_dir=parsed_args.docs_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
