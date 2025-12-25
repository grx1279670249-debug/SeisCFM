# SeisCFM

SeisCFM contains training and generation pipelines for a seismic conditional flow matching (CFM) model that predicts ground-motion spectra and reconstructs waveforms conditioned on event metadata and fault mechanisms. The repository also includes tools for nonlinear time-history analysis (NTHA) of generated motions.

## Repository layout
- `torchcfm/`: Core conditional flow matching utilities and model definitions (e.g., UNet backbones and loss functions).
- `configs/`: Training hyperparameters, dataset loaders, and visualization utilities.
- `scripts_fm/`: End-to-end scripts for training (`train_FM.py`), evaluation, quality assessment, and conditional generation (`generate.py`).
- `NTHA/`: Finite-element bridge model for nonlinear time-history analysis using OpenSeesPy.
- `Data/`: Metadata and statistics needed for dataset normalization.

## Environment setup
1. Create a Python environment (Python 3.9+ recommended).
2. Install dependencies. The main scripts rely on packages such as PyTorch, NumPy, pandas, scikit-learn, matplotlib, h5py, TensorBoard, tqdm, and OpenSeesPy (for NTHA). A minimal installation example:
   ```bash
   pip install torch numpy pandas scikit-learn matplotlib h5py tensorboard tqdm openseespy
   ```
3. Ensure a CUDA-capable GPU is available for efficient model training and sampling.

## Data preparation
1. Request and download the NGA-West2 dataset from the PEER ground-motion database: https://ngawest2.berkeley.edu/.
2. Place the downloaded `NGA_West2.hdf5` file inside the `Data/` directory. The training scripts expect this file alongside the provided `meta.csv` (event metadata) and `global_stats.npz` (normalization statistics) in `Data/`.
3. If you store the files elsewhere, update the `stats_path`, `csv_path`, and `h5_path` variables near the top of `scripts_fm/train_FM.py` and `scripts_fm/generate.py` to point to your actual locations.

## Training
Train the conditional flow matching model using the prepared NGA-West2 data. From the repository root:
```bash
python scripts_fm/train_FM.py
```
The script splits the metadata into train/validation/test subsets, logs metrics to TensorBoard, and writes checkpoints to `models/Flow_Matching_3/` by default. Edit `configs/train_parameter.py` to adjust hyperparameters such as batch size, epochs, and learning rate.

## Conditional generation
After training, you can synthesize spectra and reconstruct waveforms conditioned on magnitude, distance, Vs30, and fault type:
```bash
python scripts_fm/generate.py
```
The script loads the best checkpoint, builds conditioning batches from `meta.csv`, and saves reconstructed waveforms to the path specified by `OUT_DIR`. Update the conditioning ranges or output path in `scripts_fm/generate.py` to suit your use case.

## Nonlinear time-history analysis
Use `NTHA/FE_model.py` to perform nonlinear time-history analysis of generated motions on the included bridge model. The script demonstrates building the finite-element model with OpenSeesPy and can be adapted for your structural configurations.

## Notes
- All file paths in the scripts are plain strings; adjust them if you relocate the data or models.
- The repository assumes access to a GPU for training and generation workloads; CPU execution will be significantly slower.
