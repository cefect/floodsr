"""Build docs/user/cli_reference.rst from live CLI parser metadata."""

import argparse
from pathlib import Path

from floodsr.cli import build_parser


def _iter_documented_parsers(parser: argparse.ArgumentParser, command_path_l: list[str] | None = None):
    """Yield the parser and its subparsers in help-menu order."""
    command_path_l = command_path_l or []
    yield ("Main Command" if not command_path_l else " ".join(command_path_l), parser)

    # Walk subparsers in the same order they appear in the CLI help.
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for choice_action in action._choices_actions:
            choice_name = choice_action.dest
            yield from _iter_documented_parsers(action.choices[choice_name], [*command_path_l, choice_name])


def _format_action_label(parser: argparse.ArgumentParser, action: argparse.Action) -> str:
    """Format one action invocation for documentation output."""
    formatter = parser._get_formatter()
    return formatter._format_action_invocation(action)


def _append_definition_list(
    blocks: list[str],
    parser: argparse.ArgumentParser,
    title: str,
    item_l: list[tuple[str, str]],
):
    """Append one RST definition-list section when there are visible items."""
    if not item_l:
        return

    # Use rubrics so repeated labels like "Options" do not collide in Sphinx.
    blocks.extend([f".. rubric:: {title}", ""])
    for label, help_text in item_l:
        blocks.append(f"``{label}``")
        blocks.append(f"   {help_text}")
        blocks.append("")


def _collect_command_items(parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    """Collect subcommand names and descriptions for one parser."""
    item_l = []
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for choice_action in action._choices_actions:
            item_l.append((choice_action.dest, choice_action.help or ""))
    return item_l


def _collect_positional_items(parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    """Collect positional argument labels and help text for one parser."""
    item_l = []
    for action in parser._get_positional_actions():
        if isinstance(action, argparse._SubParsersAction):
            continue
        if action.help == argparse.SUPPRESS:
            continue
        item_l.append((_format_action_label(parser, action), action.help or ""))
    return item_l


def _collect_optional_items(parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    """Collect optional argument labels and help text for one parser."""
    item_l = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        if not action.option_strings:
            continue
        if action.help == argparse.SUPPRESS:
            continue
        item_l.append((_format_action_label(parser, action), action.help or ""))
    return item_l


def main_build_cli_reference(output_fp: Path) -> Path:
    """Generate the CLI reference page from the live parser metadata.

    Parameters
    ----------
    output_fp : Path
        Destination ``.rst`` file.

    Returns
    -------
    Path
        The written ``.rst`` path.
    """
    assert isinstance(output_fp, Path), f"output_fp must be a Path, got {type(output_fp)!r}"
    assert output_fp.suffix == ".rst", f"output_fp must end with .rst, got {output_fp}"
    parser = build_parser()

    blocks = ["CLI Reference", "=============", "", "Auto-generated from live CLI parser metadata.", ""]
    for section_name, section_parser in _iter_documented_parsers(parser):
        # Write each parser into its own readable, translatable section.
        blocks.extend([section_name, "-" * len(section_name), ""])

        if section_parser.description:
            blocks.extend([section_parser.description, ""])

        # Keep command syntax literal while exposing narrative help as normal prose.
        blocks.extend([".. rubric:: Usage", "", "::", ""])
        for line in section_parser.format_usage().rstrip("\n").splitlines():
            blocks.append(f"   {line}")
        blocks.append("")

        _append_definition_list(blocks, section_parser, "Commands", _collect_command_items(section_parser))
        _append_definition_list(blocks, section_parser, "Positional Arguments", _collect_positional_items(section_parser))
        _append_definition_list(blocks, section_parser, "Options", _collect_optional_items(section_parser))

    # Ensure the destination directory exists before writing output.
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    output_fp.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")
    return output_fp


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for CLI reference generation."""
    parser = argparse.ArgumentParser(description="Generate docs/user/cli_reference.rst from live CLI parser metadata.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/user/cli_reference.rst"),
        help="Destination RST file path.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed_args = _parse_arguments()
    main_build_cli_reference(output_fp=parsed_args.output)
