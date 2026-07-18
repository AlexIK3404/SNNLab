# Checkpoints and model snapshots / Checkpoint и model snapshot

## Checkpoint

A checkpoint contains everything required for exact continuation:

- model/network state;
- exact sample schedule and current position;
- RNG state;
- training history and homeostasis;
- dataset construction recipe.

Use **Continue run** after loading an unfinished checkpoint. Use **Extend training** only after the stored schedule is complete.

## Model snapshot

A model snapshot is inference-oriented and intentionally excludes the training schedule and full training history. It is smaller and clearer for distribution, but it is not a replacement for a checkpoint when exact continuation is required.

## Security

Both currently use Python pickle. Never load untrusted `.pkl` files.
