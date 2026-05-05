# Cleanup Record

Scope: keep the training/runtime path needed by local `configs/021201-19.yaml`.

## Validation Guard

- Added `tests/test_02120119_runtime_path.py` to cover the local 021201-19 path:
  config loading, BodyNet forward pass, graph distance matrix loading, and graph
  ranking loss smoke coverage.

## Cleanup Log

- Pending: semantic/text embedding initialization cleanup.
- Pending: embedding tracking cleanup.
- Pending: non-graph negative sampling and unused runtime config cleanup.
