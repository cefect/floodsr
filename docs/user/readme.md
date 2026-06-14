# USER documentation

- [ReadTheDocs](https://app.readthedocs.org/projects/floodsr/)
- [ReadTheDocs french project](https://app.readthedocs.org/projects/floodsr-fr/)

## Read the Docs config

- `.readthedocs.yaml`
- French translations now live under `docs/user/locale/fr/LC_MESSAGES/` so RTD and local Sphinx builds use the same plain `fr` language slug.

# LOCAL COMPILE with SPHINX

## main (english)

```bash
# 1) go to the docs source directory
cd /workspace/docs/user

# 2) verify sphinx is available in the current environment
python -m sphinx --version

# 3) force a clean html docs build from this directory, then print index.html
python -m sphinx --fresh-env -b html . "_build/manual"

# launch index.html in the default Windows browser (from WSL). if not on work-tree:
"\\wsl.localhost\Ubuntu\home\cefect\LS\09_REPOS\04_TOOLS\floodsr\docs\user\_build\manual\index.html"
```

or.. run containerized from wsl
```bash
`/home/cefect/LS/09_REPOS/04_TOOLS/floodsr/docs/user/scripts/run_sphinx_docker.sh`
```

## french

```bash
# 1) go to the docs source directory
cd /workspace/docs/user

# compile the fr .po catalogs to .mo files
bash scripts/compile_fr_catalogs.sh
 
# 3) build the fr html docs into a separate build directory
python -m sphinx -E -b html -D language=fr . "_build/fr_html"

# launch the fr index.html in the default Windows browser (from WSL)
\\wsl.localhost\Ubuntu\home\cefect\LS\09_REPOS\04_TOOLS\floodsr\docs\user\_build\fr_html\index.html
```




# MAINTAIN


## updating `docs/user/cli_reference.rst`

This page is maintained manually for docs builds.
If the CLI changes and you want to refresh the page from the live parser metadata, run:

```bash
cd /workspace
python docs/user/scripts/build_cli_reference.py
```

## update tutorial notebooks
see [docs/user/notebooks/readme.md](../user/notebooks/readme.md)

## translation maintenance

Follow [ADR-0018: Docs and Tutorials Strategy](../dev/adr/0018-docs-and-tutorials.md) for the translation architecture and review expectations.

### schema

See [ADR-0018](../dev/adr/0018-docs-and-tutorials.md) for the translation metadata schema, review-state definitions, and the `.po` entry comment example.

### workflow

Use this workflow whenever you want to bring the French docs up to date with the current English docs.

The metadata-driven stale check works from the `.po` entries themselves. 
For each entry, the sync step normalizes the current `msgid`, hashes it, and compares that value to the stored `source_hash`. 
If the values differ, the English source for that entry changed and the entry should no longer be treated as trusted.

Initial migration only:

```bash
python scripts/seed_fr_review_metadata.py
```

1. Go to the docs source directory.

   ```bash
   cd /workspace/docs/user
   ```

2. Run the translation sync script to update the relevant `.po` `msgid` values to match the current English docs, recompute `hash(normalize(msgid))`, compare that value to `source_hash`, and update the review state in place.

   Expected behavior:

   - unchanged English + `human_locked`: leave untouched
   - changed English: keep the current `msgstr` as reference and mark `stale`
   - new English entry: add an empty `msgstr` and mark `stale`

   ```bash
   python scripts/sync_fr_translations.py --repo-root /workspace
   ```

   For a scoped validation pass on one catalog before a repo-wide run, use:

   ```bash
   python scripts/sync_fr_translations.py --repo-root /workspace --catalog user_guide.po --dry-run
   ```

   By default the script writes the reviewer queue to `docs/user/_build/fr_translation_review.csv` and `docs/user/_build/fr_translation_review.md`.

3. Draft French translations only for entries marked `stale`.

   The LLM should update only `stale` entries, write or refresh the `msgstr`, and flip those entries to `llm_draft`. It should not touch `human_locked` entries.

   Agent-only trivial-change filter:

	   - If a previously `human_locked` entry changed only by a tiny English wording tweak and the existing human-reviewed French still matches the meaning, the agent may keep or restore `human_locked`.
	   - If an entry changed only because of source wrapping, gettext segmentation, or literal/code markup around an untranslated command, option, path, or code token, keep or restore `human_locked` when the rendered meaning is unchanged.
	   - Use this only for genuinely trivial edits. New sections, split or merged entries, added detail, or meaningfully revised guidance should remain `llm_draft`.
	   - This is a manual agent review step only and should not be implemented in the sync script or other automation.

4. Send only `llm_draft` entries for human review using poedit.com.

   The sync script writes review-only catalogs with `llm_draft` entries to `docs/user/_fr_review/`.
   To refresh that folder without rerunning the full sync, run:

   ```bash
   python scripts/copy_fr_review_catalogs.py
   ```

5. Reviewer reviews .po entries in poedit.com

   When a reviewer approves an entry, set `review_status` to `human_locked` and refresh `reviewed_at` and `reviewer`.

   import back into project with `docs/user/scripts/port_emma_po_updates.sh`

6. Compile the French catalogs.

   ```bash
   bash scripts/compile_fr_catalogs.sh
   ```

7. Build the French HTML and review the rendered output.

   ```bash
   python -m sphinx -E -b html -D language=fr . "_build/fr_html"
   ```

Operational notes:

- Do not infer trusted translation state from git history alone. Use the per-entry metadata comments as the primary state.
- Git history is still useful for audit trails and for linking a review packet back to the English-side commits.
- Prefer keeping the previous approved `msgstr` in place when an entry becomes `stale` so the reviewer can compare the prior trusted wording against the new English source.
- This trivial-change `human_locked` filter is an agent judgment layer applied after sync, not a rule for the scripts.
- Only touch/review `.po` files whose English source changed since the last translation refresh.

### instructions for the translator

- Target French (`fr`).
- Do not use `gettext` to generate translation files. Edit the existing `.po` catalogs directly.
- Preserve the metadata comments on each entry and update them only when the workflow explicitly changes the entry state.
- Keep commands, code, stdout, option names, paths, and project names unchanged, including `HRDEM` and `CostGrow`.
- In `cli_reference.po`, translate narrative help text and explanatory prose, but do not translate literal commands, subcommands, flags, option names, paths, or code-like tokens.
- When a term should stay tied to the English wording, explain it in French rather than forcing a literal translation. For example, `to high resolution (tohr)` should note the English phrase in the French text.
- Translate for readability and fidelity, not word-for-word mechanical alignment.
- If an entry is marked `human_locked` and the English source has not changed, do not rewrite it.
- If an entry is marked `stale`, use the previous French text as context, but align the updated translation to the current English source.
- After translation work, compile the catalogs and review the rendered French docs to confirm translation, navigation, and links behave as intended.
