"""Build docs/user/cli_reference.rst from live CLI --help output."""

import argparse, shlex, subprocess
from pathlib import Path


def main_build_cli_reference(output_fp: Path, cli_cmd: str = "floodsr") -> Path:
    """Generate the CLI reference page from command help output.

    Parameters
    ----------
    output_fp : Path
        Destination ``.rst`` file.
    cli_cmd : str, default "floodsr"
        Base command used to retrieve help text.

    Returns
    -------
    Path
        The written ``.rst`` path.
    """
    assert isinstance(output_fp, Path), f"output_fp must be a Path, got {type(output_fp)!r}"
    assert output_fp.suffix == ".rst", f"output_fp must end with .rst, got {output_fp}"
    assert cli_cmd.strip(), "cli_cmd cannot be empty"

    # Define command paths to document from live help text.
    command_groups = [
        ("Main Command", [cli_cmd, "--help"]),
        ("tohr", [cli_cmd, "tohr", "--help"]),
        ("models", [cli_cmd, "models", "--help"]),
        ("models list", [cli_cmd, "models", "list", "--help"]),
        ("models fetch", [cli_cmd, "models", "fetch", "--help"]),
        ("doctor", [cli_cmd, "doctor", "--help"]),
    ]

    blocks = ["CLI Reference", "=============", "", "Auto-generated from live command help output.", ""]
    for section_name, cmd in command_groups:
        # Capture help output directly from the command path.
        run_result = subprocess.run(cmd, text=True, capture_output=True)
        if run_result.returncode != 0 and "No such file or directory" in (run_result.stderr or ""):
            # Fallback to module invocation if floodsr binary is unavailable.
            fallback_cmd = ["python", "-m", "floodsr.cli"] + cmd[1:]
            run_result = subprocess.run(fallback_cmd, text=True, capture_output=True)
            cmd = fallback_cmd
        if run_result.returncode != 0:
            raise RuntimeError(
                f"failed help command `{shlex.join(cmd)}` with code {run_result.returncode}: {run_result.stderr}"
            )

        # Write each command output into its own readable section.
        blocks.extend(
            [
                section_name,
                "-" * len(section_name),
                "",
                ".. code-block:: text",
                "",
            ]
        )
        for line in run_result.stdout.rstrip("\n").splitlines():
            blocks.append(f"   {line}")
        blocks.append("")

    # Ensure the destination directory exists before writing output.
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    output_fp.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")
    return output_fp


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for CLI reference generation."""
    parser = argparse.ArgumentParser(description="Generate docs/user/cli_reference.rst from live CLI help.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/user/cli_reference.rst"),
        help="Destination RST file path.",
    )
    parser.add_argument(
        "--cli-cmd",
        type=str,
        default="floodsr",
        help="CLI executable name to use for help capture.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed_args = _parse_arguments()
    main_build_cli_reference(output_fp=parsed_args.output, cli_cmd=parsed_args.cli_cmd)
