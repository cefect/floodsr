"""Seed French docs review metadata from the last trusted baseline commit."""

import argparse, subprocess
from pathlib import Path

from fr_translation_utils import (
    PoEntry,
    align_entry_blocks,
    build_review_rows,
    clone_entry,
    compute_source_hash,
    entry_has_translation,
    parse_po_catalog,
    render_po_catalog,
    set_entry_metadata,
    write_review_reports,
)


def _read_git_file(repo_dir: Path, git_ref: str, rel_path: Path) -> str | None:
    """Read one tracked file at a given git ref."""
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "show", f"{git_ref}:{rel_path.as_posix()}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def _apply_new_entry_metadata(entry: PoEntry) -> str:
    """Apply metadata for a post-baseline entry."""
    status = "llm_draft" if entry_has_translation(entry) else "stale"
    set_entry_metadata(entry, review_status=status, source_hash=compute_source_hash(entry))
    return status


def _apply_seed_metadata(
    baseline_entry_l: list[PoEntry],
    current_entry_l: list[PoEntry],
    reviewed_at: str,
    reviewer: str,
) -> tuple[list[PoEntry], dict[str, int]]:
    """Apply baseline review metadata to one current entry list."""
    stats_d = {"human_locked": 0, "stale": 0, "llm_draft": 0}
    out_entry_l = []
    for tag, old_block, new_block in align_entry_blocks(baseline_entry_l, current_entry_l):
        if tag == "equal":
            for entry in new_block:
                new_entry = clone_entry(entry)
                set_entry_metadata(
                    new_entry,
                    review_status="human_locked",
                    source_hash=compute_source_hash(new_entry),
                    reviewed_at=reviewed_at,
                    reviewer=reviewer,
                )
                out_entry_l.append(new_entry)
                stats_d["human_locked"] += 1
            continue

        pair_count = min(len(old_block), len(new_block))
        for old_entry, new_entry in zip(old_block[:pair_count], new_block[:pair_count], strict=False):
            stale_entry = clone_entry(new_entry)
            set_entry_metadata(
                stale_entry,
                review_status="stale",
                source_hash=compute_source_hash(old_entry),
                reviewed_at=reviewed_at,
                reviewer=reviewer,
            )
            out_entry_l.append(stale_entry)
            stats_d["stale"] += 1

        for entry in new_block[pair_count:]:
            draft_entry = clone_entry(entry)
            stats_d[_apply_new_entry_metadata(draft_entry)] += 1
            out_entry_l.append(draft_entry)
    return out_entry_l, stats_d


def main_seed_fr_review_metadata(
    repo_dir: Path,
    catalog_root: Path,
    baseline_ref: str,
    reviewed_at: str,
    reviewer: str,
    report_dir: Path,
    catalog_rel_l: list[str] | None = None,
) -> dict[str, int]:
    """Seed review metadata across the French catalogs from a trusted baseline.

    Parameters
    ----------
    repo_dir : Path
        Repository root used for `git show`.
    catalog_root : Path
        Root directory containing French `.po` catalogs.
    baseline_ref : str
        Git ref for the last fully trusted baseline.
    reviewed_at : str
        ISO date written to `human_locked` entries.
    reviewer : str
        Reviewer name or id written to `human_locked` entries.
    report_dir : Path
        Directory receiving the review queue reports.
    catalog_rel_l : list[str] | None
        Optional subset of catalog paths relative to `catalog_root`.

    Returns
    -------
    dict[str, int]
        Summary counts for the seeded catalogs.
    """
    assert repo_dir.is_dir(), f"missing repo_dir:\n    {repo_dir}"
    assert catalog_root.is_dir(), f"missing catalog_root:\n    {catalog_root}"
    assert baseline_ref.strip(), "baseline_ref must be non-empty"
    assert reviewed_at.strip(), "reviewed_at must be non-empty"
    assert reviewer.strip(), "reviewer must be non-empty"

    stats_d = {"catalogs": 0, "human_locked": 0, "stale": 0, "llm_draft": 0}
    review_row_l = []
    po_fp_l = (
        [catalog_root / rel_path for rel_path in sorted(catalog_rel_l)]
        if catalog_rel_l
        else sorted(catalog_root.rglob("*.po"))
    )

    # Seed each tracked French catalog against its baseline counterpart.
    for po_fp in po_fp_l:
        rel_path = po_fp.relative_to(repo_dir)
        current_catalog = parse_po_catalog(po_fp.read_text(encoding="utf-8"))
        baseline_text = _read_git_file(repo_dir, baseline_ref, rel_path)
        if baseline_text is None:
            for entry in current_catalog.entry_l:
                stats_d[_apply_new_entry_metadata(entry)] += 1
            po_fp.write_text(render_po_catalog(current_catalog), encoding="utf-8")
            stats_d["catalogs"] += 1
            review_row_l.extend(build_review_rows(str(po_fp.relative_to(catalog_root)), current_catalog.entry_l))
            continue

        baseline_catalog = parse_po_catalog(baseline_text)
        current_catalog.entry_l, catalog_stats_d = _apply_seed_metadata(
            baseline_catalog.entry_l,
            current_catalog.entry_l,
            reviewed_at=reviewed_at,
            reviewer=reviewer,
        )
        po_fp.write_text(render_po_catalog(current_catalog), encoding="utf-8")
        stats_d["catalogs"] += 1
        for key in ("human_locked", "stale", "llm_draft"):
            stats_d[key] += catalog_stats_d[key]
        review_row_l.extend(build_review_rows(str(po_fp.relative_to(catalog_root)), current_catalog.entry_l))

    write_review_reports(
        review_row_l,
        csv_fp=report_dir / "fr_translation_review.csv",
        markdown_fp=report_dir / "fr_translation_review.md",
    )
    return stats_d


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for metadata seeding."""
    parser = argparse.ArgumentParser(description="Seed review metadata in French docs catalogs from a baseline git ref.")
    parser.add_argument("--repo-dir", type=Path, default=Path(__file__).resolve().parents[3], help="Repository root.")
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "locale" / "fr" / "LC_MESSAGES",
        help="French catalog root.",
    )
    parser.add_argument(
        "--baseline-ref",
        default="0fe2c899498af12fd65692cb75a2d808d4bd70b5",
        help="Git ref for the last trusted French baseline.",
    )
    parser.add_argument("--reviewed-at", default="2026-03-29", help="ISO date for the trusted baseline review.")
    parser.add_argument("--reviewer", default="Emma H", help="Reviewer id/name for the trusted baseline.")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "_build",
        help="Directory for markdown/csv review reports.",
    )
    parser.add_argument("catalogs", nargs="*", help="Optional catalog paths relative to the French catalog root.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed_args = _parse_arguments()
    stats = main_seed_fr_review_metadata(
        repo_dir=parsed_args.repo_dir,
        catalog_root=parsed_args.catalog_root,
        baseline_ref=parsed_args.baseline_ref,
        reviewed_at=parsed_args.reviewed_at,
        reviewer=parsed_args.reviewer,
        report_dir=parsed_args.report_dir,
        catalog_rel_l=parsed_args.catalogs or None,
    )
    print(stats)
