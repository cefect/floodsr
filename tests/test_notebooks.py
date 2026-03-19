"""Notebook execution tests for `docs/user/notebooks`.

Run these from the `dev` conda environment with:
`conda run -n dev pytest -q -m "notebook" tests/test_notebooks.py`
"""

import os, pathlib, shutil, subprocess

import pytest


@pytest.mark.parametrize(
    "notebook_fp",
    [
        pytest.param(pathlib.Path("docs/user/notebooks/tutorial_1.ipynb"), id="tutorial_1"),
        pytest.param(pathlib.Path("docs/user/notebooks/tutorial_2.ipynb"), id="tutorial_2"),
    ],
)
@pytest.mark.network
@pytest.mark.notebook
def test_tutorial_notebook_executes(notebook_fp, tmp_path):
    """Execute each tutorial notebook from a temporary copy and confirm it produced outputs."""
    nbformat = pytest.importorskip("nbformat", reason="notebook tests require the dev conda environment")
    assert notebook_fp.exists(), f"missing tutorial notebook:\n    {notebook_fp}"

    # Copy the source notebook so execution outputs never modify the tracked file.
    run_fp = tmp_path / notebook_fp.name
    shutil.copy2(notebook_fp, run_fp)

    # Expose the local package and a CLI shim without requiring notebook source edits.
    cli_fp = tmp_path / "floodsr"
    cli_fp.write_text("#!/usr/bin/env bash\npython -m floodsr.cli \"$@\"\n", encoding="utf-8")
    cli_fp.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    env["PYTHONPATH"] = f"{pathlib.Path.cwd()}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    notebook_tmp_dir = pathlib.Path.cwd() / "_cache" / "notebook_tmp" / notebook_fp.stem
    notebook_tmp_dir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(notebook_tmp_dir)
    env["TEMP"] = str(notebook_tmp_dir)
    env["TMP"] = str(notebook_tmp_dir)

    # Execute the temporary copy in place so relative notebook outputs stay isolated in tmp_path.
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=600",
            str(run_fp),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )

    # Re-read the executed copy and confirm the notebook structure and outputs are non-empty.
    executed_nb = nbformat.read(run_fp, as_version=4)
    assert isinstance(executed_nb, nbformat.NotebookNode)
    assert any(cell.get("outputs") for cell in executed_nb.cells if cell.get("cell_type") == "code")
