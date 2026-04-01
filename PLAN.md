# Plan

## migrate french docs to per-entry `.po` review metadata

### goal

Move from the current direct-edit `.po` workflow to a metadata-backed workflow where each French translation entry records whether it is trusted human-reviewed text, machine draft text, or stale relative to the current English source.

### baseline

Use commit `0fe2c899498af12fd65692cb75a2d808d4bd70b5` (`FR Docs update (#51)`) as the last human-proofed French baseline.

- Every translation entry present and approved at that commit should be initialized as `human_locked`.
- Any translation entry whose English source changed after that commit should no longer be treated as trusted at head.
- New or machine-updated entries after that baseline should be marked `llm_draft` or `stale` depending on whether a draft French `msgstr` already exists.

### migration steps

1. Confirm the initial schema and vocabulary.
   - Use `review_status` with only `human_locked`, `llm_draft`, and `stale`.
   - Use `source_hash` as the required machine-readable link to the normalized entry `msgid`.
   - Treat `reviewed_at` and `reviewer` as required provenance fields for entries promoted to `human_locked`.

2. Add metadata comments to all existing French `.po` entries.
   - Start from the French catalogs as they existed at commit `0fe2c899498af12fd65692cb75a2d808d4bd70b5`.
   - Seed every entry from that baseline as `human_locked`.
   - Write `reviewed_at` using the baseline commit date and `reviewer` as `Emma H`.
   - Compute and write `source_hash` for each entry from the normalized baseline `msgid`.

3. Add a developer sync script under `docs/user/scripts/`.
   - Read the English docs and update the relevant `.po` entry `msgid` values to match the current English source.
   - Use the existing 1:1 `.rst` to `.po` mapping as the file alignment rule.
   - Do not introduce `source_file` or `source_key` metadata unless the sync logic proves they are needed later.
   - Read the French `.po` catalogs.
   - Parse the per-entry metadata comments.
   - Recompute normalized hashes from the current entry `msgid`.
   - Mark entries `stale` when `hash(normalize(msgid))` differs from the stored `source_hash`.
   - Report new entries that need French text and review.
   - Emit markdown or CSV listing only `stale` and `llm_draft` entries, including the English text and current French text so a human reviewer only sees net-new review work.

4. Validate the workflow on `user_guide.po`.
   - Apply the baseline metadata to `user_guide.po`.
   - Run the sync script and confirm that changed entries move to `stale` based on `msgid` hash mismatch.
   - Run the LLM pass only on `stale` entries and flip those entries to `llm_draft`.
   - Confirm that unchanged baseline entries remain `human_locked`.

5. Roll out the migration across all French catalogs.
   - Backfill baseline metadata across the remaining French `.po` files.
   - Run the sync script repo-wide to update `msgid` values and mark changed or new entries as `stale`.
   - Run the LLM pass only on `stale` entries so the migrated repo lands in a mixed `human_locked` and `llm_draft` state.
   - Promote entries back to `human_locked` only after human review.

6. Compile and validate the French docs build.
   - Compile the `.po` catalogs to `.mo`.
   - Build the French Sphinx HTML successfully.
   - Review the rendered output for obvious translation, navigation, and link regressions.

7. Finalize the steady-state process documentation.
   - Keep the ADR limited to the architectural decisions.
   - Keep the developer workflow and translator instructions in `docs/user/readme.md`.

 
 
