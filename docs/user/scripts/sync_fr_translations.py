"""Sync French docs catalogs against current English sources and review metadata."""

import argparse, csv, hashlib, io, json, logging, shutil, subprocess, sys, tempfile
from pathlib import Path

from babel.messages import pofile
from babel.messages.catalog import Catalog, Message


BASELINE_COMMIT = "0fe2c899498af12fd65692cb75a2d808d4bd70b5"
BASELINE_REVIEWED_AT = "2026-03-29"
BASELINE_REVIEWER = "Emma H"
REVIEW_STATUSES = {"human_locked", "llm_draft", "stale"}
METADATA_KEYS = ("review_status", "source_hash", "reviewed_at", "reviewer")


def main_sync_fr_translations(
    repo_root: Path,
    catalog_rel_l: list[str] | None = None,
    baseline_commit: str = BASELINE_COMMIT,
    report_dir: Path | None = None,
    review_dir: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Sync French catalogs to current docs sources and refresh review metadata.

    Parameters
    ----------
    repo_root : Path
        Repository root containing ``docs/user`` and ``.git``.
    catalog_rel_l : list[str] | None, default=None
        Optional relative ``.po`` paths beneath ``locale/fr/LC_MESSAGES`` to sync.
    baseline_commit : str, default=BASELINE_COMMIT
        Git commit used as the last fully human-reviewed French baseline.
    report_dir : Path | None, default=None
        Directory for the reviewer report artifacts. Defaults to ``docs/user/_build``.
    review_dir : Path | None, default=None
        Directory for copies of catalogs with ``llm_draft`` entries. Defaults to ``docs/user/_fr_review``.
    dry_run : bool, default=False
        When ``True``, do not write catalogs or reports.

    Returns
    -------
    dict
        Summary stats and report paths.
    """
    assert isinstance(repo_root, Path), f"repo_root must be a Path, got {type(repo_root)!r}"
    assert repo_root.exists(), f"missing repo root:\n    {repo_root}"
    assert (repo_root / ".git").exists(), f"missing git metadata under:\n    {repo_root}"
    docs_dir = repo_root / "docs" / "user"
    catalog_root = docs_dir / "locale" / "fr" / "LC_MESSAGES"
    report_dir = report_dir or docs_dir / "_build"
    review_dir = review_dir or docs_dir / "_fr_review"
    assert docs_dir.exists(), f"missing docs dir:\n    {docs_dir}"
    assert catalog_root.exists(), f"missing fr catalog dir:\n    {catalog_root}"

    log = logging.getLogger(__name__)
    log.info(f"syncing fr catalogs from\n    {catalog_root}")
    log.debug(
        json.dumps(
            {
                "baseline_commit": baseline_commit,
                "catalogs": catalog_rel_l or "ALL",
                "dry_run": dry_run,
                "report_dir": str(report_dir),
                "review_dir": str(review_dir),
            },
            indent=2,
        )
    )

    stats = {"files": 0, "entries": 0, "human_locked": 0, "llm_draft": 0, "stale": 0}
    report_row_l = []
    with tempfile.TemporaryDirectory(prefix="floodsr_fr_sync_") as temp_dir:
        template_root = _run_gettext_build(docs_dir=docs_dir, temp_dir=Path(temp_dir))
        for po_fp in _iter_catalog_fp_l(catalog_root=catalog_root, catalog_rel_l=catalog_rel_l):
            file_stats, file_row_l = _sync_one_catalog(
                repo_root=repo_root,
                po_fp=po_fp,
                catalog_root=catalog_root,
                template_root=template_root,
                baseline_commit=baseline_commit,
                dry_run=dry_run,
            )
            stats["files"] += 1
            stats["entries"] += file_stats["entries"]
            stats["human_locked"] += file_stats["human_locked"]
            stats["llm_draft"] += file_stats["llm_draft"]
            stats["stale"] += file_stats["stale"]
            report_row_l.extend(file_row_l)

    report_path_d = _write_review_reports(report_dir=report_dir, report_row_l=report_row_l, dry_run=dry_run)
    stats.update(report_path_d)
    review_path_d = copy_review_catalogs(
        catalog_root=catalog_root,
        review_dir=review_dir,
        report_row_l=report_row_l,
        dry_run=dry_run,
    )
    stats.update(review_path_d)
    log.info(
        f"synced {stats['files']} catalog(s), {stats['entries']} entry(ies), "
        f"{stats['human_locked']} human_locked, {stats['llm_draft']} llm_draft, {stats['stale']} stale"
    )
    return stats


def copy_review_catalogs(catalog_root: Path, review_dir: Path, report_row_l: list[dict], dry_run: bool = False) -> dict:
    """Write review-only catalogs with ``llm_draft`` entries into the human review folder."""
    assert catalog_root.exists(), f"missing catalog root:\n    {catalog_root}"
    review_dir = review_dir.resolve()
    catalog_rel_l = sorted({row["catalog"] for row in report_row_l if row.get("status") == "llm_draft"})
    if dry_run:
        return {"review_dir": str(review_dir), "review_catalogs": len(catalog_rel_l)}

    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    for catalog_rel in catalog_rel_l:
        source_fp = catalog_root / catalog_rel
        target_fp = review_dir / catalog_rel
        assert source_fp.exists(), f"missing review catalog source:\n    {source_fp}"
        review_catalog = _build_llm_draft_catalog(source_catalog=_read_po_catalog(po_fp=source_fp, locale="fr"))
        assert any(message.id for message in review_catalog), f"no llm_draft entries found in:\n    {source_fp}"
        target_fp.parent.mkdir(parents=True, exist_ok=True)
        target_fp.write_text(_render_po_catalog_text(current_catalog=review_catalog), encoding="utf-8")
    return {"review_dir": str(review_dir), "review_catalogs": len(catalog_rel_l)}


def _build_llm_draft_catalog(source_catalog: Catalog) -> Catalog:
    """Build a catalog containing only entries still flagged for human review."""
    review_catalog = Catalog(
        locale=source_catalog.locale,
        project=source_catalog.project,
        version=source_catalog.version,
        copyright_holder=source_catalog.copyright_holder,
        msgid_bugs_address=source_catalog.msgid_bugs_address,
        creation_date=source_catalog.creation_date,
        revision_date=source_catalog.revision_date,
        last_translator=source_catalog.last_translator,
        language_team=source_catalog.language_team,
        charset=source_catalog.charset,
        fuzzy=False,
    )
    for message in source_catalog:
        if not message.id:
            continue
        metadata = _parse_review_metadata(message=message)
        if metadata.get("review_status") != "llm_draft":
            continue
        review_catalog.add(
            message.id,
            string=message.string,
            locations=message.locations,
            flags=message.flags,
            auto_comments=message.auto_comments,
            user_comments=message.user_comments,
            previous_id=message.previous_id,
            lineno=message.lineno,
            context=message.context,
        )
    return review_catalog


def _run_gettext_build(docs_dir: Path, temp_dir: Path) -> Path:
    """Build current gettext templates for the user docs into a temp directory."""
    assert docs_dir.exists(), f"missing docs dir:\n    {docs_dir}"
    assert temp_dir.exists(), f"missing temp dir:\n    {temp_dir}"
    template_root = temp_dir / "gettext"
    cmd = [sys.executable, "-m", "sphinx", "-q", "-b", "gettext", str(docs_dir), str(template_root)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Sphinx gettext build failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return template_root


def _iter_catalog_fp_l(catalog_root: Path, catalog_rel_l: list[str] | None) -> list[Path]:
    """Resolve the ordered list of French ``.po`` files to process."""
    assert catalog_root.exists(), f"missing catalog root:\n    {catalog_root}"
    if catalog_rel_l:
        return [catalog_root / rel_path for rel_path in sorted(catalog_rel_l)]
    return sorted(catalog_root.rglob("*.po"))


def _sync_one_catalog(
    repo_root: Path,
    po_fp: Path,
    catalog_root: Path,
    template_root: Path,
    baseline_commit: str,
    dry_run: bool,
) -> tuple[dict, list[dict]]:
    """Sync one French catalog against the current English template and metadata."""
    assert po_fp.suffix == ".po", f"expected .po catalog, got:\n    {po_fp}"
    original_text = po_fp.read_text(encoding="utf-8")
    rel_path = po_fp.relative_to(catalog_root)
    pot_fp = template_root / rel_path.with_suffix(".pot")
    assert pot_fp.exists(), f"missing gettext template for catalog:\n    {pot_fp}"

    # Read the current, baseline, and template catalogs before reconciling state.
    current_catalog = _read_po_catalog(po_fp=po_fp, locale="fr")
    current_meta_d = {
        (message.context, message.id): {
            "metadata": _parse_review_metadata(message=message),
            "auto_comments": _strip_metadata_comments(message.auto_comments),
        }
        for message in current_catalog
        if message.id
    }
    current_catalog.update(_read_po_catalog(po_fp=pot_fp, locale=None), update_creation_date=True)
    baseline_catalog = _read_baseline_catalog(repo_root=repo_root, baseline_commit=baseline_commit, rel_path=rel_path)
    baseline_msg_d = {(message.context, message.id): message for message in baseline_catalog if message.id}

    stats = {"entries": 0, "human_locked": 0, "llm_draft": 0, "stale": 0}
    report_row_l = []
    for message in current_catalog:
        if not message.id:
            continue
        stats["entries"] += 1

        # Re-attach stored metadata/comments after Babel updates the msgids.
        current_key = (message.context, message.id)
        previous_key = (message.context, _coerce_previous_id(message.previous_id)) if message.previous_id else None
        previous_payload = current_meta_d.get(previous_key) if previous_key else None
        current_payload = current_meta_d.get(current_key)
        metadata = dict((current_payload or previous_payload or {}).get("metadata") or {})
        extra_auto_comments = list((current_payload or previous_payload or {}).get("auto_comments") or [])

        # Drop fuzzy so review state is carried by explicit metadata instead.
        message.flags.discard("fuzzy")
        source_hash = _build_source_hash(message_id=message.id)
        baseline_message = baseline_msg_d.get(current_key)
        message.string, review_status, reviewed_at, reviewer = _resolve_message_state(
            message=message,
            baseline_message=baseline_message,
            metadata=metadata,
            source_hash=source_hash,
        )

        metadata_lines = [f"review_status: {review_status}", f"source_hash: {source_hash}"]
        if review_status == "human_locked":
            metadata_lines.extend([f"reviewed_at: {reviewed_at}", f"reviewer: {reviewer}"])
        message.auto_comments = metadata_lines + extra_auto_comments
        stats[review_status] += 1
        if review_status in {"stale", "llm_draft"}:
            report_row_l.append(
                {
                    "catalog": rel_path.as_posix(),
                    "status": review_status,
                    "english": _render_message_id(message.id),
                    "french": _render_message_string(message.string),
                    "previous_english": _render_message_id(_coerce_previous_id(message.previous_id)),
                    "locations": _format_locations(message.locations),
                }
            )

    # Skip write-back when the only diff is volatile header churn.
    rendered_text = _render_po_catalog_text(current_catalog=current_catalog)
    final_text = _prune_catalog_headers_text(text=rendered_text)
    if _strip_volatile_po_headers(original_text) == _strip_volatile_po_headers(final_text):
        return stats, report_row_l

    # Write the reconciled catalog after all entry metadata has been refreshed.
    if not dry_run:
        po_fp.parent.mkdir(parents=True, exist_ok=True)
        po_fp.write_text(final_text, encoding="utf-8")
    return stats, report_row_l


def _resolve_message_state(
    message: Message,
    baseline_message: Message | None,
    metadata: dict,
    source_hash: str,
) -> tuple[str | tuple[str, ...] | list[str], str, str | None, str | None]:
    """Resolve message text and review state for one synced catalog entry."""
    assert message.id, "message.id is required"
    assert source_hash, "source_hash is required"
    review_status = metadata.get("review_status")
    stored_source_hash = metadata.get("source_hash")

    # Honor previously migrated metadata when the current English source hash still matches.
    if review_status in REVIEW_STATUSES and stored_source_hash:
        if stored_source_hash == source_hash:
            reviewed_at = metadata.get("reviewed_at") if review_status == "human_locked" else None
            reviewer = metadata.get("reviewer") if review_status == "human_locked" else None
            if review_status == "human_locked" and baseline_message and not _message_has_text(message):
                return baseline_message.string, review_status, reviewed_at, reviewer
            if review_status != "human_locked" or (reviewed_at and reviewer):
                return message.string, review_status, reviewed_at, reviewer
        return message.string, "stale", None, None

    # Seed the initial migration from the last human-reviewed baseline when possible.
    if baseline_message and (not _message_has_text(message) or message.string == baseline_message.string):
        return baseline_message.string, "human_locked", BASELINE_REVIEWED_AT, BASELINE_REVIEWER

    # Keep existing draft French text when present; otherwise leave the entry stale for drafting.
    if _message_has_text(message):
        return message.string, "llm_draft", None, None
    return message.string, "stale", None, None


def _read_po_catalog(po_fp: Path, locale: str | None) -> Catalog:
    """Read one ``.po`` or ``.pot`` catalog with Babel."""
    assert po_fp.exists(), f"missing catalog path:\n    {po_fp}"
    with po_fp.open(encoding="utf-8") as stream:
        return pofile.read_po(stream, locale=locale)


def _read_baseline_catalog(repo_root: Path, baseline_commit: str, rel_path: Path) -> Catalog:
    """Read one baseline French catalog from git, or return an empty catalog if missing."""
    assert repo_root.exists(), f"missing repo root:\n    {repo_root}"
    git_rel_path = Path("docs") / "user" / "locale" / "fr" / "LC_MESSAGES" / rel_path
    cmd = ["git", "-C", str(repo_root), "show", f"{baseline_commit}:{git_rel_path.as_posix()}"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return Catalog(locale="fr")
    return pofile.read_po(result.stdout.splitlines(True), locale="fr")


def _render_po_catalog_text(current_catalog: Catalog) -> str:
    """Render one Babel catalog to the normalized on-disk PO text format."""
    stream = io.BytesIO()
    pofile.write_po(
        stream,
        current_catalog,
        ignore_obsolete=True,
        include_previous=False,
        no_location=True,
        sort_by_file=True,
        width=100,
    )
    return stream.getvalue().decode("utf-8")


def _prune_catalog_headers_text(text: str) -> str:
    """Remove non-project PO header fields we do not want to track."""
    drop_prefix_l = (
        "\"Report-Msgid-Bugs-To:",
        "\"Last-Translator:",
        "\"Generated-By:",
    )
    line_l = text.splitlines()
    kept_line_l = [line for line in line_l if not line.startswith(drop_prefix_l)]
    return "\n".join(kept_line_l).rstrip() + "\n"


def _strip_volatile_po_headers(text: str) -> str:
    """Normalize volatile PO header lines before comparing tracked file content."""
    volatile_prefix_l = ("\"POT-Creation-Date:",)
    line_l = _prune_catalog_headers_text(text=text).splitlines()
    kept_line_l = [line for line in line_l if not line.startswith(volatile_prefix_l)]
    return "\n".join(kept_line_l).rstrip() + "\n"


def _parse_review_metadata(message: Message) -> dict:
    """Parse translator-comment review metadata from one PO message."""
    metadata = {}
    for comment in message.auto_comments:
        if ":" not in comment:
            continue
        key, value = comment.split(":", 1)
        key = key.strip()
        if key in METADATA_KEYS:
            metadata[key] = value.strip()
    return metadata


def _strip_metadata_comments(auto_comment_l: list[str]) -> list[str]:
    """Return auto comments with only review metadata comments removed."""
    cleaned_comment_l = []
    for comment in auto_comment_l:
        if ":" in comment and comment.split(":", 1)[0].strip() in METADATA_KEYS:
            continue
        cleaned_comment_l.append(comment)
    return cleaned_comment_l


def _normalize_message_id(message_id) -> str:
    """Normalize one gettext message id into a stable hash input string."""
    if isinstance(message_id, (list, tuple)):
        return "\x1f".join(_normalize_message_id(part) for part in message_id)
    if message_id is None:
        return ""
    return " ".join(str(message_id).split())


def _build_source_hash(message_id) -> str:
    """Build the normalized source hash recorded in review metadata."""
    return hashlib.sha256(_normalize_message_id(message_id).encode("utf-8")).hexdigest()[:8]


def _message_has_text(message: Message) -> bool:
    """Return whether the message currently carries any French translation text."""
    if isinstance(message.string, (list, tuple)):
        return any(str(part).strip() for part in message.string)
    return bool(str(message.string).strip())


def _coerce_previous_id(previous_id):
    """Coerce Babel ``previous_id`` values into the same shape as ``message.id``."""
    if not previous_id:
        return None
    if isinstance(previous_id, tuple):
        return previous_id
    if isinstance(previous_id, list):
        return previous_id[0] if len(previous_id) == 1 else tuple(previous_id)
    return previous_id


def _render_message_id(message_id) -> str:
    """Render a gettext msgid payload into a single report-friendly string."""
    if message_id is None:
        return ""
    if isinstance(message_id, (list, tuple)):
        return " | ".join(str(part) for part in message_id)
    return str(message_id)


def _render_message_string(message_string) -> str:
    """Render a gettext msgstr payload into a single report-friendly string."""
    if isinstance(message_string, (list, tuple)):
        return " | ".join(str(part) for part in message_string if part)
    return str(message_string or "")


def _write_review_reports(report_dir: Path, report_row_l: list[dict], dry_run: bool) -> dict:
    """Write CSV and Markdown review packets for non-human-locked entries."""
    report_dir = report_dir.resolve()
    csv_fp = report_dir / "fr_translation_review.csv"
    md_fp = report_dir / "fr_translation_review.md"
    if dry_run:
        return {"report_csv": str(csv_fp), "report_md": str(md_fp)}

    report_dir.mkdir(parents=True, exist_ok=True)
    with csv_fp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["catalog", "status", "english", "french", "previous_english", "locations"],
        )
        writer.writeheader()
        writer.writerows(report_row_l)

    md_line_l = ["# French Translation Review Packet", ""]
    if not report_row_l:
        md_line_l.extend(["No `stale` or `llm_draft` entries were found.", ""])
    else:
        for index, row in enumerate(report_row_l, start=1):
            md_line_l.extend(
                [
                    f"## {index}. {row['catalog']} [{row['status']}]",
                    "",
                    f"- Locations: {row['locations'] or 'n/a'}",
                    f"- Previous English: {row['previous_english'] or 'n/a'}",
                    "",
                    "### English",
                    "",
                    row["english"] or "(empty)",
                    "",
                    "### French",
                    "",
                    row["french"] or "(empty)",
                    "",
                ]
            )
    md_fp.write_text("\n".join(md_line_l), encoding="utf-8")
    return {"report_csv": str(csv_fp), "report_md": str(md_fp)}


def _format_locations(location_l: list[tuple[str, int | None]]) -> str:
    """Render only real source locations for the review packet."""
    cleaned_location_l = [f"{path}:{lineno}" for path, lineno in location_l if lineno is not None]
    return "; ".join(cleaned_location_l)


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the French translation sync."""
    parser = argparse.ArgumentParser(description="Sync docs/user fr catalogs against current English docs sources.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root containing docs/user and .git.",
    )
    parser.add_argument(
        "--catalog",
        action="append",
        default=[],
        help="Relative .po path beneath docs/user/locale/fr/LC_MESSAGES to sync. Repeat as needed.",
    )
    parser.add_argument(
        "--baseline-commit",
        default=BASELINE_COMMIT,
        help="Git commit used as the last fully human-reviewed French baseline.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "_build",
        help="Output directory for the review CSV/Markdown artifacts.",
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "_fr_review",
        help="Output directory for .po catalogs that still contain llm_draft entries.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute updates without writing catalogs or reports.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed_args = _parse_arguments()
    logging.basicConfig(level=logging.DEBUG if parsed_args.debug else logging.INFO, format="%(message)s")
    summary = main_sync_fr_translations(
        repo_root=parsed_args.repo_root.resolve(),
        catalog_rel_l=parsed_args.catalog or None,
        baseline_commit=parsed_args.baseline_commit,
        report_dir=parsed_args.report_dir.resolve(),
        review_dir=parsed_args.review_dir.resolve(),
        dry_run=parsed_args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
