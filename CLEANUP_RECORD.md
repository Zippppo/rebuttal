# Cleanup Record

Scope: keep the training/runtime path needed by local `configs/021201-19.yaml`.

## Validation Guard

- Added `tests/test_02120119_runtime_path.py` to cover the local 021201-19 path:
  config loading, BodyNet forward pass, graph distance matrix loading, and graph
  ranking loss smoke coverage.

## Cleanup Log

- Done: removed semantic/text embedding initialization from the 021201 runtime
  model path. Label embedding directions are now always random, matching the
  local `configs/021201-19.yaml` behavior. Deprecated config keys
  `hyp_direction_mode` and `hyp_text_embedding_path` are ignored silently so
  the local ignored config remains runnable.
- Pending: embedding tracking cleanup.
- Pending: non-graph negative sampling and unused runtime config cleanup.
