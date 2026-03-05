## Patent Translation Batch Runner

### Run all documents

```bash
uv run python main.py
```

This runs the full batch from the configured S3 prefix and saves outputs to `outputs/`.

### Enable source-vs-translation compare reports

```bash
uv run python main.py --compare
```

When enabled, each translated document produces:

- `outputs/comparisons/<docname>.compare.chunks.md` (chunk-level source vs translation)
- `outputs/comparisons/<docname>.compare.lines.tsv` (line-level mapping by block id)

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
