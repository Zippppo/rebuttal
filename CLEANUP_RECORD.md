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
- Done: removed optional embedding tracking from training. The 021201 config does
  not enable embedding visualization, so the training path no longer imports the
  tracker or computes system labels for tracker output.
- Done: narrowed training negative sampling to the graph-distance matrix path
  required by 021201. Removed the runtime mode switch and unused spatial
  adjacency config fields from `Config`; deprecated keys are ignored silently
  for the local ignored config file.
- Done: removed unused LR scheduler branches and the unused `num_levels`
  config. Training now constructs the 021201 multiphase cosine scheduler
  directly while ignoring deprecated scheduler keys from the local config.
