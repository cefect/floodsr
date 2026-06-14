"""Notebook execution tests for `docs/user/notebooks`.

Run these from the `dev` conda environment with:

conda run -n dev pytest -q -m "notebook" tests/test_tutorials.py
"""

import json, os, pathlib, shutil, subprocess

import pytest

@pytest.mark.parametrize(
    "notebook_fp,timeout_s,shared_cache_dir",
    [
        pytest.param(pathlib.Path("docs/user/notebooks/tutorial_1.ipynb"), 600, None, id="tutorial_1"),
        pytest.param(pathlib.Path("docs/user/notebooks/tutorial_2.ipynb"), 600, None, id="tutorial_2"),
        pytest.param(pathlib.Path("docs/user/notebooks/tutorial_3.ipynb"), 3600, pathlib.Path.cwd() / "_cache", id="tutorial_3"),
        pytest.param(pathlib.Path("docs/user/notebooks/tutorial_4.ipynb"), 1200, None, id="tutorial_4"),
    ],
)
@pytest.mark.network
@pytest.mark.notebook
def test_tutorial_notebook_executes(notebook_fp, timeout_s, shared_cache_dir, tmp_path, capsys):
    """Execute each tutorial notebook from a temporary copy and confirm it produced outputs."""
    nbformat = pytest.importorskip("nbformat", reason="notebook tests require the dev conda environment")
    assert notebook_fp.exists(), f"missing tutorial notebook:\n    {notebook_fp}"

    # Fail early on deprecated CRS examples so notebook runs stop with a clear message.
    source_text = json.dumps(nbformat.read(notebook_fp, as_version=4))
    assert "EPSG:3778" not in source_text, (
        f"deprecated CRS EPSG:3778 found in notebook source:\n    {notebook_fp}\n"
        "use a supported CRS such as EPSG:3978 instead"
    )

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
    notebook_tmp_dir = tmp_path / "tmp"
    notebook_tmp_dir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(notebook_tmp_dir)
    env["TEMP"] = str(notebook_tmp_dir)
    env["TMP"] = str(notebook_tmp_dir)
    if shared_cache_dir is not None:
        shared_cache_dir = pathlib.Path(shared_cache_dir).resolve()
        shared_cache_dir.mkdir(parents=True, exist_ok=True)
        env["FLOODSR_SHARED_CACHE_DIR"] = str(shared_cache_dir)

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout_s}",
        str(run_fp),
    ]
    with capsys.disabled():
        print(
            f"notebook test inputs:\n"
            f"    source={notebook_fp}\n"
            f"    run_fp={run_fp}\n"
            f"    tmp_path={tmp_path}\n"
            f"    tmpdir={notebook_tmp_dir}\n"
            f"    shared_cache_dir={env.get('FLOODSR_SHARED_CACHE_DIR', 'none')}\n"
            f"    cmd={' '.join(cmd)}",
            flush=True,
        )

    # Execute the temporary copy in place so relative notebook outputs stay isolated in tmp_path.
    proc = subprocess.Popen(
        cmd,
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    stdout_l = []
    assert proc.stdout is not None
    for line in proc.stdout:
        stdout_l.append(line)
        with capsys.disabled():
            print(line, end="", flush=True)
    proc.wait()
    stdout = "".join(stdout_l)
    if proc.returncode != 0:
        pytest.fail(
            f"notebook execution failed with code={proc.returncode}\n"
            f"    source={notebook_fp}\n"
                f"    run_fp={run_fp}\n"
                f"    tmp_path={tmp_path}\n"
                f"    tmpdir={notebook_tmp_dir}\n"
                f"    shared_cache_dir={env.get('FLOODSR_SHARED_CACHE_DIR', 'none')}\n"
                f"    cmd={' '.join(cmd)}\n"
                f"stdout:\n{stdout}"
        )

    # Re-read the executed copy and confirm the notebook structure and outputs are non-empty.
    executed_nb = nbformat.read(run_fp, as_version=4)
    assert isinstance(executed_nb, nbformat.NotebookNode)
    assert any(cell.get("outputs") for cell in executed_nb.cells if cell.get("cell_type") == "code")
