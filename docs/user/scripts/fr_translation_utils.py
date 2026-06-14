"""Utilities for French docs translation catalog maintenance."""

import ast, csv, hashlib, json, re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path


REVIEW_STATUS_VALUES = {"human_locked", "llm_draft", "stale"}
METADATA_KEY_L = ("review_status", "source_hash", "reviewed_at", "reviewer")
METADATA_PATTERN = re.compile(r"^#\.\s*(review_status|source_hash|reviewed_at|reviewer):\s*(.*?)\s*$")


@dataclass
class PoEntry:
    """Represent one editable PO entry block."""

    comment_l: list[str] = field(default_factory=list)
    msgctxt: str | None = None
    msgid: str | None = None
    msgid_plural: str | None = None
    msgstr_d: dict[int, str] = field(default_factory=dict)
    msgstr_order_l: list[int] = field(default_factory=list)


@dataclass
class PoCatalog:
    """Represent one PO catalog with a raw header block."""

    header_l: list[str]
    entry_l: list[PoEntry]


def split_po_blocks(text: str) -> list[list[str]]:
    """Split a PO file into logical blocks."""
    block_l, lines = [], []
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


def parse_po_catalog(text: str) -> PoCatalog:
    """Parse a PO catalog into a header block and entry objects."""
    block_l = split_po_blocks(text)
    assert block_l, "PO catalog is empty"
    assert any(line.startswith("msgid") for line in block_l[0]), "PO catalog header block is missing msgid"
    entry_l = []
    for block in block_l[1:]:
        entry = parse_po_entry(block)
        if entry.msgid is None:
            continue
        entry_l.append(entry)
    return PoCatalog(header_l=block_l[0], entry_l=entry_l)


def parse_po_entry(lines: list[str]) -> PoEntry:
    """Parse one PO entry block."""
    entry = PoEntry()
    field_name = None
    field_index = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            entry.comment_l.append(line)
            continue
        if stripped.startswith("msgctxt"):
            field_name = "msgctxt"
            field_index = None
            entry.msgctxt = ast.literal_eval(stripped[len("msgctxt") :].strip())
            continue
        if stripped.startswith("msgid_plural"):
            field_name = "msgid_plural"
            field_index = None
            entry.msgid_plural = ast.literal_eval(stripped[len("msgid_plural") :].strip())
            continue
        if stripped.startswith("msgid"):
            field_name = "msgid"
            field_index = None
            entry.msgid = ast.literal_eval(stripped[len("msgid") :].strip())
            continue
        if stripped.startswith("msgstr["):
            match = re.match(r"msgstr\[(\d+)\]\s+(.*)", stripped)
            assert match, f"failed to parse plural msgstr line: {stripped!r}"
            field_name = "msgstr"
            field_index = int(match.group(1))
            entry.msgstr_d[field_index] = ast.literal_eval(match.group(2))
            if field_index not in entry.msgstr_order_l:
                entry.msgstr_order_l.append(field_index)
            continue
        if stripped.startswith("msgstr"):
            field_name = "msgstr"
            field_index = 0
            entry.msgstr_d[0] = ast.literal_eval(stripped[len("msgstr") :].strip())
            if 0 not in entry.msgstr_order_l:
                entry.msgstr_order_l.append(0)
            continue
        if stripped.startswith('"'):
            value = ast.literal_eval(stripped)
            if field_name == "msgctxt":
                entry.msgctxt = (entry.msgctxt or "") + value
            elif field_name == "msgid":
                entry.msgid = (entry.msgid or "") + value
            elif field_name == "msgid_plural":
                entry.msgid_plural = (entry.msgid_plural or "") + value
            elif field_name == "msgstr":
                assert field_index is not None, "msgstr continuation missing index"
                entry.msgstr_d[field_index] = entry.msgstr_d.get(field_index, "") + value
            continue
    return entry


def render_po_catalog(catalog: PoCatalog) -> str:
    """Render a parsed PO catalog back to text."""
    block_l = ["\n".join(catalog.header_l)]
    for entry in catalog.entry_l:
        block_l.append(render_po_entry(entry))
    return "\n\n".join(block_l).rstrip() + "\n"


def render_po_entry(entry: PoEntry) -> str:
    """Render one PO entry block."""
    assert entry.msgid is not None, "PO entry missing msgid"
    line_l = [*entry.comment_l]
    if entry.msgctxt is not None:
        line_l.append(f"msgctxt {json.dumps(entry.msgctxt, ensure_ascii=False)}")
    line_l.append(f"msgid {json.dumps(entry.msgid, ensure_ascii=False)}")
    if entry.msgid_plural is not None:
        line_l.append(f"msgid_plural {json.dumps(entry.msgid_plural, ensure_ascii=False)}")
    if entry.msgid_plural is None and entry.msgstr_order_l in ([], [0]):
        line_l.append(f"msgstr {json.dumps(entry.msgstr_d.get(0, ''), ensure_ascii=False)}")
        return "\n".join(line_l)
    for index in entry.msgstr_order_l or sorted(entry.msgstr_d):
        line_l.append(f"msgstr[{index}] {json.dumps(entry.msgstr_d.get(index, ''), ensure_ascii=False)}")
    return "\n".join(line_l)


def parse_entry_metadata(entry: PoEntry) -> tuple[dict[str, str], list[str]]:
    """Split metadata comments from other entry comments."""
    meta_d, comment_l = {}, []
    for line in entry.comment_l:
        match = METADATA_PATTERN.match(line)
        if match:
            meta_d[match.group(1)] = match.group(2)
            continue
        comment_l.append(line)
    return meta_d, comment_l


def set_entry_metadata(
    entry: PoEntry,
    review_status: str,
    source_hash: str,
    reviewed_at: str | None = None,
    reviewer: str | None = None,
) -> None:
    """Replace metadata comments for one entry."""
    assert review_status in REVIEW_STATUS_VALUES, f"invalid review_status: {review_status!r}"
    assert isinstance(source_hash, str) and source_hash.strip(), "source_hash must be a non-empty string"
    assert review_status != "human_locked" or (reviewed_at and reviewer), (
        "human_locked entries require reviewed_at and reviewer"
    )
    _, comment_l = parse_entry_metadata(entry)
    meta_l = [
        f"#. review_status: {review_status}",
        f"#. source_hash: {source_hash}",
    ]
    if reviewed_at:
        meta_l.append(f"#. reviewed_at: {reviewed_at}")
    if reviewer:
        meta_l.append(f"#. reviewer: {reviewer}")
    entry.comment_l = [*meta_l, *comment_l]


def normalize_entry_msgid(entry: PoEntry) -> str:
    """Normalize an entry msgid for stable change detection."""
    assert entry.msgid is not None, "PO entry missing msgid"
    raw = entry.msgid if entry.msgid_plural is None else "\n".join([entry.msgid, entry.msgid_plural])
    return re.sub(r"\s+", " ", raw.strip())


def compute_source_hash(entry: PoEntry) -> str:
    """Compute the compact normalized source hash for one entry."""
    normalized = normalize_entry_msgid(entry)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]


def entry_has_translation(entry: PoEntry) -> bool:
    """Return whether an entry has any non-empty French text."""
    return any(value.strip() for value in entry.msgstr_d.values())


def clone_entry(entry: PoEntry) -> PoEntry:
    """Clone one PO entry for safe editing."""
    return PoEntry(
        comment_l=[*entry.comment_l],
        msgctxt=entry.msgctxt,
        msgid=entry.msgid,
        msgid_plural=entry.msgid_plural,
        msgstr_d={**entry.msgstr_d},
        msgstr_order_l=[*entry.msgstr_order_l],
    )


def align_entry_blocks(old_entry_l: list[PoEntry], new_entry_l: list[PoEntry]) -> list[tuple[str, list[PoEntry], list[PoEntry]]]:
    """Align two entry lists by normalized msgid sequence."""
    old_key_l = [normalize_entry_msgid(entry) for entry in old_entry_l]
    new_key_l = [normalize_entry_msgid(entry) for entry in new_entry_l]
    matcher = SequenceMatcher(a=old_key_l, b=new_key_l, autojunk=False)
    return [(tag, old_entry_l[i1:i2], new_entry_l[j1:j2]) for tag, i1, i2, j1, j2 in matcher.get_opcodes()]


def build_review_rows(catalog_rel: str, entry_l: list[PoEntry]) -> list[dict[str, str]]:
    """Build review-report rows for non-finalized entries."""
    row_l = []
    for entry in entry_l:
        meta_d, _ = parse_entry_metadata(entry)
        review_status = meta_d.get("review_status", "")
        if review_status not in {"stale", "llm_draft"}:
            continue
        row_l.append(
            {
                "catalog": catalog_rel,
                "review_status": review_status,
                "source_hash": meta_d.get("source_hash", ""),
                "reviewed_at": meta_d.get("reviewed_at", ""),
                "reviewer": meta_d.get("reviewer", ""),
                "msgid": entry.msgid or "",
                "msgstr": entry.msgstr_d.get(0, ""),
            }
        )
    return row_l


def write_review_reports(row_l: list[dict[str, str]], csv_fp: Path, markdown_fp: Path) -> None:
    """Write CSV and Markdown review reports."""
    assert csv_fp.suffix == ".csv", f"expected csv path, got {csv_fp}"
    assert markdown_fp.suffix == ".md", f"expected markdown path, got {markdown_fp}"
    csv_fp.parent.mkdir(parents=True, exist_ok=True)
    markdown_fp.parent.mkdir(parents=True, exist_ok=True)
    field_l = ["catalog", "review_status", "source_hash", "reviewed_at", "reviewer", "msgid", "msgstr"]

    with csv_fp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_l)
        writer.writeheader()
        writer.writerows(row_l)

    line_l = [
        "# French translation review queue",
        "",
        f"Entries requiring follow-up: {len(row_l)}",
        "",
    ]
    for row in row_l:
        line_l.extend(
            [
                f"## {row['catalog']}",
                "",
                f"- review_status: `{row['review_status']}`",
                f"- source_hash: `{row['source_hash']}`",
                f"- reviewed_at: `{row['reviewed_at'] or '-'}'",
                f"- reviewer: `{row['reviewer'] or '-'}'",
                "",
                "### English",
                "",
                row["msgid"] or "_empty_",
                "",
                "### French",
                "",
                row["msgstr"] or "_empty_",
                "",
            ]
        )
    markdown_fp.write_text("\n".join(line_l).rstrip() + "\n", encoding="utf-8")
