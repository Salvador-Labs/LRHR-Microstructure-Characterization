# LR2HR: local microstructure estimation

Code and original Lightning checkpoints accompanying *Deep Learning Enhanced Estimation of Local Three-Phase Microstructural Characteristics from Ideal 2D or Low-Resolution 3D Data* (Kent et al.). This is a prepared publication package; no repository URL or final paper DOI has been assigned here.

## Quick start

Use a fresh Python 3.10 or 3.11 environment. From this directory:

```sh
python -m pip install -e .
python predict.py examples/inputs --output predictions.csv
python validate_examples.py
```

CPU inference is the default; `--device cuda` or `--device mps` selects an available accelerator. `predict.py` writes physical descriptors and a separate normalized-output CSV. No training data are needed to load a checkpoint. Run any script with `--help` for its complete CLI.

Inputs are synthetic XCT intensity arrays. A denotes anode contrast (phase intensities 13500, 43500, 55500); C uses 13500, 43500, 76500. The loader normalizes these ranges internally. Use `--normalized` only for inputs already scaled to [0, 1]. The default example has three 24³ arrays at 0.5 µm, derived from 248³ segmented volumes at 0.05 µm. Integer block averaging uses the first 240³ voxels, dropping the trailing eight voxels on each axis. Exact agreement with archived predictions has not been established; the bundled examples support evaluation against their stored ground truth. See docs/validation.md.

## Model bundles

| Model | Input shape (including channel) | Source run | Best epoch (zero-based) |
|---|---|---|---|
| `lrhr_0.3um_A` | 1 × 41 × 41 × 41 | version_74 | 3372 |
| `lrhr_0.3um_C` | 1 × 41 × 41 × 41 | version_73 | 2465 |
| `lrhr_0.5um_A` | 1 × 24 × 24 × 24 | version_75 | 6448 |
| `lrhr_0.5um_C` | 1 × 24 × 24 × 24 | version_72 | 7043 |
| `lrhr_0.75um_A` | 1 × 16 × 16 × 16 | version_78 | 14348 |
| `lrhr_0.75um_C` | 1 × 16 × 16 × 16 | version_79 | 12039 |
| `lrhr_1.5um_A` | 1 × 8 × 8 × 8 | version_76 | 16819 |
| `lrhr_1.5um_C` | 1 × 8 × 8 × 8 | version_71 | 19471 |
| `lrhr_1um_A` | 1 × 12 × 12 × 12 | version_68 | 17974 |
| `lrhr_1um_C` | 1 × 12 × 12 × 12 | version_69 | 16903 |
| `lrhr_2um_A` | 1 × 6 × 6 × 6 | version_77 | 19957 |
| `lrhr_2um_C` | 1 × 6 × 6 × 6 | version_70 | 18637 |

Select a nondefault bundle with `python predict.py INPUT_DIRECTORY --model models/MODEL_ID --output predictions.csv`. Each bundle contains the unchanged original `model.ckpt`, its SHA-256, explicit `config.json`, matching `normalization.csv`, and original training hyperparameters for provenance. The original Lightning version was 1.9.3; loading with 2.3 upgrades metadata in memory. Do not run a checkpoint migration over the original assets.

```python
from lvclr import load_bundle
model, config = load_bundle('models/lrhr_0.5um_A')
# model is a PyTorch Lightning module in evaluation mode.
```

For direct Lightning loading, pass the parsed config JSON as `model_config` to `PublishedModel.load_from_checkpoint`. Never mix a checkpoint and normalization from different bundles.

## Inputs, outputs, and examples

`prepare_inputs.py VOLUME --output-dir prepared` creates inputs from a segmented reference volume (`.npy`, or `.npz` with `phase_labels`). The selected model determines shape and preprocessing. It does not segment experimental images or simulate their noise/point-spread function.

The 15 physical output columns are phase fractions phi1–phi3, tortuosity factors tau1–tau3, particle diameters d1–d3 (µm), connected TPB density (µm⁻²), connected/total TPB fraction, total TPB density (µm⁻²), and interfacial areas S12/S13/S23 (µm⁻¹). Inverse transforms are implemented in `src/lvclr/model.py`; the paper's factor-of-two TPB conversion and 0.05 µm reference voxel size are retained. Predictions are not clipped to physical bounds.

`examples/provenance.json` records sample identities and processing. Reference targets and expected predictions are supplied separately. These examples overlap historical training data: they validate execution, **not held-out accuracy**. Expected predictions were generated with this release and are not a substitute for independently reproducing archived predictions. See [validation details](docs/validation.md).

## Training

Full training data are intentionally excluded. The underlying synthetic microstructure bank is hosted by NETL EDX: [dataset record](https://doi.org/10.18141/1988063). Obtain the required inputs and matching descriptor labels externally. The release does not claim that the original model-ready HDF5 files or exact split manifests are contained in that deposit.

Create an explicit manifest with `input`, `split` (`train` or `val`), and 15 **normalized** target columns:
`f1,f2,f3,tort1,tort2,tort3,davg1,davg2,davg3,tpbconn,tpbconnfrac,tpbtot,sa12,sa13,sa23`.
Normalization must match the selected bundle. For HDF5 inputs include `h5_index`; LRHR additionally requires `job_group` and rejects job groups shared between splits.

`prepare_training_manifest.py` can join existing labeled inputs to an explicit split table (columns `name,split,job_group`). For 2D input directories provide `--targets` and use `<volume>_x/y/z.npy` names. For LRHR supply HDF5 with `X`, `y`, and `names`; `y` must already be normalized in the listed order. It does not invent historical memberships.

```sh
python train.py training_manifest.csv --output-dir runs/retrain
python train.py training_manifest.csv --output-dir runs/check --fast-dev-run --accelerator cpu
```

Default architecture, shape, learning rate, epoch count, batch size, seed and augmentation come from `models/lrhr_0.5um_A/config.json`. Use `--config models/MODEL_ID/config.json` to change model. Defaults retain Adam at constant 1e-4, batch size 32, and 20000 epochs. The effective configuration and manifest are saved for each run. New training uses explicit seeded augmentation and split manifests; historical unseeded preprocessing and unavailable memberships prevent a claim of bitwise retraining reproducibility. Historical 2D flips also apply during validation; LRHR applies fixed fourfold augmentation to each split independently.

Resume a checkpoint created by this release with `--resume runs/retrain/training/version_0/checkpoints/last.ckpt` and a higher total `--epochs`. Original paper checkpoints are supplied as inference assets, not supported training-resume inputs. `predict.py --checkpoint NEW_CHECKPOINT --model MATCHING_BUNDLE` supports new checkpoints with matching preprocessing and normalization.

## Layout and license

- `src/`: architecture, Lightning wrapper, preprocessing and output conversion.
- `models/`: model bundles and machine-readable manifest.
- `examples/`: small software-validation examples and reference outputs.
- `train.py`, `predict.py`: training and inference entry points.
- `docs/validation.md`: checks and remaining provenance limitations.

Code is provided under the [MIT license](LICENSE). External NETL datasets retain their own terms and citation requirements. Cite the associated paper and the dataset when using them.

## Normalized HDF5 performance example

Run `python evaluate_hdf5.py` to evaluate all 16 cases in `examples/test_subvolumes_0.5um_A_normalized.h5`. Outputs are physical-unit predictions, ground truth and per-property MAE/RMSE/maximum errors in `example_evaluation/`.

`X` contains float32 arrays of shape (16,24,24,24), already normalized as `(raw - 13500)/42000`. **Do not apply that normalization again.** `y` contains the original 15 standardized transformed descriptors, cast to float32 as in the original loader. `names` preserves sample IDs. Dataset attributes describe the model and column order. The evaluation script selects the matching checkpoint and conversion file automatically.

All 16 names share the bedcbcbccaaz0 prefix and differ in their final subvolume suffix. They span a narrow range of structures and are related examples, not a broad independent test set. Property ranges are recorded in `examples/example_microstructure_variability.csv`.
