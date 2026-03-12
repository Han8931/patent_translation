## Patent Translation Batch Runner

### Run all documents

```bash
uv run python main.py
```

This runs the full batch from the configured S3 prefix and saves outputs to `outputs/`.

### Enable source-vs-output compare reports

```bash
uv run python main.py --compare
```

When enabled, each translated document produces:

- `outputs/comparisons/<docname>.compare.chunks.md`
- `outputs/comparisons/<docname>.compare.lines.tsv`

Comparison is built from the actual DOCX files:

- Source DOCX (downloaded original)
- Output DOCX (final translated file in `outputs/`)

Report details:

- Chunk-level source/output text comparison
- Line-level comparison by block ID (`p:*`, `cell:*`)
- Unmatched block detection:
  - missing in output
  - extra in output
- TSV `match_status` values:
  - `matched`
  - `missing_in_output`
  - `extra_in_output`

You can combine retry + compare:

```bash
uv run python main.py --retry_failed --compare
```

### Compare only (reuse existing translated DOCX)

If you already have a translated DOCX, generate compare reports without rerunning translation:

```bash
uv run python main.py --compare_only --src_docx <source.docx> --translated_docx <translated.docx>
```

This still writes:

- `outputs/comparisons/<docname>.compare.chunks.md`
- `outputs/comparisons/<docname>.compare.lines.tsv`

### Retry only failed documents

```bash
uv run python main.py --retry_failed
```

Default failed-list path:

- `outputs/failed_keys.txt`

You can override it:

```bash
uv run python main.py --retry_failed --failed_list outputs/failed_keys.txt
```

### Failed list behavior

After each run, the final failed document keys are written to `outputs/failed_keys.txt`.

The failed list includes:

- S3 download failures
- LLM translation failures
- Documents with unresolved translation ID mismatches (unexpected/missing IDs after retries)
