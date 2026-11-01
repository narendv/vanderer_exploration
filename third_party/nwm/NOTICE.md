# Notice

This folder contains code adapted from [NoMaD World Model](https://github.com/facebookresearch/nwm), commit `3f6cd8e70d6f2d1e2b9684acff510710135f0f41`, by Meta Platforms, Inc. and affiliates. It's licensed under CC BY-NC 4.0 (see [LICENSE.md](LICENSE.md)).

Only the files needed for fine-tuning and inference are included.

## Changes from upstream

- `datasets.py`, `train.py`: added `waypoint_spacing`, which samples frames every N steps and scales goal distances and action normalisation to match.
- `isolated_nwm_infer.py`: evaluation no longer requires predefined index files, and fixed the `init_distributed` call.
- All files: moved into the `nwm` package, with imports and config paths updated to match.
- `config/data_config.yaml`: changed `action_stats` and added CARLA datasets.
- `config/eval_config.yaml`: changed dataset paths.
- `config/nwm_cdit_b_finetune.yaml`: added.
