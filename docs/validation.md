# Example data and validation checks

The examples demonstrate how to load an LRHR model, run inference, and compare predictions with stored ground-truth descriptors. They are related synthetic subvolumes that overlap historical training data, not an independent validation benchmark or a representative sample of the full microstructure library.

## Evaluate the HDF5 example

After installing the repository dependencies, run this command from the repository root:

```bash
python evaluate_hdf5.py
```

The script loads `models/lrhr_0.5um_A/` and evaluates all 16 cases in `examples/test_subvolumes_0.5um_A_normalized.h5`. It writes the following files to `example_evaluation/`:

- `predictions.csv`: predicted descriptors in physical units.
- `ground_truth.csv`: corresponding ground-truth descriptors in physical units.
- `errors.csv`: mean absolute error (MAE), root mean squared error (RMSE), and maximum absolute error for each descriptor, calculated over the 16 cases.

To choose another output directory:

```bash
python evaluate_hdf5.py --output-dir my_example_results
```

### Data and normalization

The HDF5 file contains:

| Dataset | Shape | Contents |
|---|---|---|
| `X` | `(16, 24, 24, 24)` | Pre-normalized, single-channel input volumes stored as float32 |
| `y` | `(16, 15)` | Standardized, transformed ground-truth descriptors stored as float32 |
| `names` | `(16,)` | Sample filenames, aligned with `X` and `y` |

Each input has a voxel size of 0.5 µm and uses synthetic anode contrast. The raw phase intensities are 13,500, 43,500, and 55,500, with intermediate intensities produced by downsampling. The following input normalization has **already been applied**:

```text
X_normalized = (X_raw - 13500) / 42000
```

Do not normalize these HDF5 inputs a second time. `evaluate_hdf5.py` passes them to the model as pre-normalized data and adds the channel dimension required by the network.

The targets use a separate normalization: property-specific transformations followed by standardization. The evaluation script uses the matching `models/lrhr_0.5um_A/normalization.csv` to undo those operations for both predictions and targets before calculating errors. Input intensity normalization must not be applied to the targets.

The target column order is:

```text
f1, f2, f3, tort1, tort2, tort3, davg1, davg2, davg3,
tpbconn, tpbconnfrac, tpbtot, sa12, sa13, sa23
```

The physical outputs comprise dimensionless volume fractions, tortuosity factors and connected/total TPB fraction; particle diameters in µm; TPB densities in µm⁻²; and interfacial areas per unit volume in µm⁻¹. Conversion retains the paper's 0.05 µm high-resolution reference voxel size and factor-of-two TPB convention.

### Scope of the examples

All 16 sample names share the `bedcbcbccaaz0` prefix and differ in their final subvolume suffix. They span a narrow range of microstructures. Ground-truth ranges and sample standard deviations are provided in `examples/example_microstructure_variability.csv`. Errors on this small, related set should not be interpreted as generalization performance across the full dataset or experimental images.

The separate `.npy` examples in `examples/inputs/` contain three **raw-intensity** volumes. Unlike the pre-normalized HDF5 inputs, these require the default normalization applied by `predict.py`:

```bash
python predict.py examples/inputs --output predictions.csv
```

Do not use `--normalized` for those raw `.npy` examples.

## Software checks and reference predictions

Run the following commands from the repository root:

```bash
python validate_examples.py
python -m unittest discover -s tests
```

`validate_examples.py` checks bundled checkpoint and normalization-file hashes and compares predictions for the three `.npy` examples with reference outputs generated using this release. The unit tests check input encoding and rejection of invalid inputs.

The twelve bundled LRHR checkpoints were loaded with strict parameter matching and produced finite 15-component outputs on CPU. Local checks also exercised example inference, training on a small synthetic manifest, checkpoint saving, and training resume. These checks verify software execution; they do not establish scientific accuracy or reproduce full training runs. They were performed with PyTorch 2.3.0.post100, PyTorch Lightning 2.3.0, and NumPy 1.26.0. Fresh-environment installation and GPU execution have not been tested as part of these checks.

## Training data

Full training data are not included. See the README for the NETL EDX dataset reference and training instructions. Model configurations and normalization files accompany each checkpoint. Retraining requires an explicit input/target split manifest; exact historical split memberships and stochastic preprocessing have not been reconstructed.
