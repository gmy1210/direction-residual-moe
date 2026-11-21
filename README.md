# Directional Residual Dual-Expert Model

DirEMoE is a binary classifier for channel-aligned brain recordings. It fuses a temporal expert based on frozen MantisV2 embeddings with a spatial expert based on Pearson functional connectivity. The spatial expert adds a learned correction from signed-lag connectivity to its full-recording static representation. A sample-wise router combines the two experts' logits.

## Installation

```bash
python -m pip install '.[mantis]'
```

Use Python 3.10 or later. The implementation was verified with PyTorch 2.6 and `mantis-tsfm` 1.0.0. For training and evaluation from cached features, `python -m pip install .` is sufficient. Signal extraction and inference use the upstream [MantisV2 weights](https://huggingface.co/paris-noah/MantisV2). Use `--weights` or the Python `weights` argument to point to an existing local copy.

## Input data

Supply a numeric NPZ with these arrays.

| Key | Shape | Meaning |
|---|---|---|
| `signals` | `[N,C,T]` | Preprocessed recordings in a consistent channel or ROI order |
| `lengths` | `[N]` | Optional integer observed lengths for right-padded recordings. Defaults to `T` |
| `labels` | `[N]` | Binary labels encoded as 0 and 1 |

Use CC200 ROI series for REST-MDD and ABIDE. Preserve native sampling and ROI order. The extractor interpolates to 512 samples for MantisV2 only. Connectivity uses the original observed samples.

Supply one split NPZ per fold with nonempty, disjoint integer vectors named `train`, `val` and `test`. These are zero-based row indices into the signals NPZ and must match its row order. Keep each subject within a single partition per fold. Use split indices matching the recording order. REST-MDD uses site-aware five-fold splitting, and ABIDE uses five site-stratified holdouts. ABIDE test subjects can recur across holdouts.

The package contains source code and benchmark settings. Dataset acquisition, upstream preprocessing, split arrays and model weights are supplied separately.

## Dataset acquisition

- [REST-MDD](docs/datasets/rest-mdd/README.md) covers access, the study cohort and CC200 ROI extraction.
- [ABIDE I](docs/datasets/abide/README.md) covers the phenotype table, QC-871 and CC200 downloads.

For either dataset, transpose each time-by-ROI matrix to `[200, T_i]`, right-pad to the longest recording in the dataset, and store the original lengths alongside the padded arrays. Save numeric arrays `signals`, `lengths`, and `labels` in an NPZ as described above. Keep a subject/site table in the same row order when building split indices. Then run `diremoe extract` with `--dataset rest_mdd` or `--dataset abide`.

## Train and evaluate

```bash
diremoe extract --dataset rest_mdd --input data/rest_mdd.npz --output features/rest_mdd --device cuda
diremoe train --features features/rest_mdd --split data/split0.npz \
  --checkpoint runs/rest_mdd/fold0.pt --seed 42 --device cuda
diremoe evaluate --features features/rest_mdd --split data/split0.npz \
  --checkpoint runs/rest_mdd/fold0.pt --device cuda
```

Extract once per dataset. For the five splits, train each fold with seed `42 + fold` and a separate checkpoint path. Use `--device cpu` to run without CUDA. Feature extraction defaults to eight recordings per batch. The feature store keeps Pearson views and regional temporal features in float16 and mean temporal features in float32. Training reads the store in batches.

| Dataset | Channels | Temporal readout | Signed lags in samples | Validation selection | Label smoothing |
|---|---:|---|---|---|---:|
| REST-MDD | 200 | Mean | 0, ±1, ±2 | ACC | 0 |
| ABIDE | 200 | Mean | 0, ±1, ±2 | AUC | 0.1 |

Training defaults to 100 epochs. Set the training duration with `--epochs`. All presets use hidden width 32, residual bottleneck 8, regional projection width 8, dropout 0.3, and an initial temporal routing prior of 0.1. The training recipe uses AdamW, batch size 32, learning rate 0.001, weight decay 0.0005 on the expert and router weights, five warmup epochs followed by cosine decay, and an auxiliary expert loss weighted by 0.2. Feature normalization is fitted on training rows. Checkpoint selection breaks validation-score ties by lower NLL. The optional scalar bias is fitted on validation margins. Evaluation returns raw and bias-calibrated ACC, AUC and NLL.

## Physical-time lags and grouped readout

For acquisition-aware delays, include one positive sampling interval per recording in the numeric NPZ field `tr_seconds`. Add `--physical-lags 2 4` to the extraction command for 2 s and 4 s delays. Add `--regional` to retain node embeddings for temporal pooling. The estimator uses a reflected 16-tap Lanczos shift at fractional frames and native Pearson at integer frames. Each window needs at least 16 native overlapping samples.

During training, `--temporal-groups groups.npy` and `--direction-groups groups.npy` independently enable pooling. Use contiguous zero-based group IDs in node order. Fit functional group assignments on training subjects for each fold. Inference restores assignments from the checkpoint. Use `predict(signals, lengths=lengths, tr_seconds=[2.0, 2.5])` for two recordings with these sampling intervals. The feature configuration records physical delays in `lags_seconds` and uses paired signed-lag slots for view indexing.

## Inference from recordings

```python
from diremoe.predict import Predictor

predict = Predictor("runs/rest_mdd/fold0.pt", device="cuda")
output = predict(signals, lengths=lengths)
probability = output["probability_class1"]
routing_weights = output["weights"]  # temporal, spatial
```

For one recording, pass a `[C,T]` array. For a batch, use `[B,C,T]` and pass `lengths` when padded. Pass `include_short_static=False` to skip the two short-window zero-lag views. Those slots are then false in the returned `valid` mask. The classifier uses the full-recording static view and nonzero-lag pairs. `calibrated_probability_class1` applies the saved validation bias. The loader accepts `base` checkpoints matching the REST-MDD and ABIDE feature configurations.

## Connectivity convention

Window fractions are 1, 1/2 and 1/4 of the observed duration. Each scale averages its window-level Pearson matrices with a half-window stride and includes the final possible window. For positive lag `d`, an edge is `corr(x_i[t+d], x_j[t])` for `i < j`. The negative-lag view uses the transposed matrix. Each directional view is half the difference between its positive- and negative-lag views. These edges measure asymmetric lagged association. A view needs at least 16 overlapping samples. A missing static view routes the recording to the temporal expert.
