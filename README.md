# SolarFlowRefiner

**Refinement-Aware Flow Matching for Surface Solar Radiation Downscaling**

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/AAAI-2027-green.svg" alt="AAAI 2027"></a>
</p>

---

## Architecture

<p align="center">
  <img src="Figures/fig01.jpg" width="95%" alt="SolarFlowRefiner Architecture"/>
</p>

**Figure 1**: Overview of SolarFlowRefiner. ERA5 radiative variables and SolarCube auxiliary channels condition a FlowMatch residual generator, which predicts an initial normalized residual relative to the upsampled ERA5 SSR baseline. A PDE-style refiner then performs iterative correction updates over K=8 refinement levels before reconstructing the high-resolution SSR field. In SolarFlowRefiner, the refinement loss is backpropagated through both the refiner and the FlowMatch sampler, making generation refinement-aware.

---

## Overview

High-resolution surface solar radiation (SSR) is critical for solar forecasting and grid operation. However, coarse reanalysis products like ERA5 (~0.25° resolution) cannot resolve the sharp, localized irradiance changes caused by clouds. This work addresses three key challenges in SSR downscaling:

1. **One-to-many ambiguity**: A single coarse ERA5 cell may contain both sunlit and cloud-shadowed regions, making the reconstruction inherently uncertain.

2. **Oversmoothing**: Deterministic models trained with pixel-wise losses tend to regress toward conditional means, missing sharp cloud-boundary gradients.

3. **Stage-wise mismatch**: Post-hoc refinement optimizes the generator independently from the correction process, even though the generator output determines the refiner's initial state.

**SolarFlowRefiner** addresses these issues through refinement-aware generation. A conditional FlowMatch model first generates a normalized SSR residual, then a prediction-conditioned PDE-style refiner iteratively corrects structured errors. Crucially, the refinement loss is backpropagated through the differentiable FlowMatch sampler, allowing the generator to learn residuals that are both accurate and refinable.

---

## Key Results

### Test-Set Performance

All metrics computed after reconstructing physical SSR from predicted normalized residuals. Lower is better for MAE, RMSE, LPIPS, FID; higher is better for SSIM.

| Method | MAE ↓ | RMSE ↓ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|--------|--------|---------|---------|----------|--------|
| Bicubic | 137.22 | 169.61 | 0.339 | 0.664 | 290.15 |
| EDSR | 29.53 | 42.71 | 0.675 | 0.187 | 46.03 |
| RCAN | 14.81 | 24.11 | 0.724 | 0.177 | 44.92 |
| SwinIR | 16.71 | 26.91 | 0.712 | 0.183 | 39.21 |
| HAT | 15.25 | 24.07 | 0.726 | 0.171 | 37.73 |
| ResShift | 13.94 | 20.68 | 0.725 | 0.159 | 36.47 |
| FlowMatch | 14.88 | 23.30 | 0.742 | 0.154 | **35.42** |
| FlowRefiner-ODE | 14.60 | 22.85 | 0.743 | 0.156 | 36.43 |
| FlowRefiner-PDE | 13.58 | 20.31 | 0.727 | 0.161 | 37.49 |
| **SolarFlowRefiner** | **11.13** | **17.11** | **0.827** | **0.137** | 36.23 |

**Improvements over FlowRefiner-PDE (post-hoc baseline):**
- MAE: 13.58 → 11.13 (**~18% reduction**)
- RMSE: 20.31 → 17.11 (**~16% reduction**)
- SSIM: 0.727 → 0.827 (**+13.7% structural fidelity**)

### Qualitative Comparison

<p align="center">
  <img src="Figures/fig02.png" width="95%" alt="Qualitative SSR Reconstruction"/>
</p>

**Figure 2**: SSR reconstruction and error maps for a held-out test sample. SolarFlowRefiner produces the lowest per-sample MAE and RMSE, reducing both broad residual bias and localized cloud-edge errors.

### Input-Channel Ablation

The ERA5 SSR baseline is used for residual reconstruction in all settings; ablation changes only the conditioning channels provided to the learned generator and refiner.

| Conditioning | MAE ↓ | RMSE ↓ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|--------------|--------|---------|---------|----------|--------|
| ERA5 only | 21.82 | 36.21 | 0.651 | 0.172 | 41.90 |
| SolarCube only | 17.31 | 24.70 | 0.710 | 0.155 | 37.10 |
| **ERA5 + SolarCube** | **11.13** | **17.11** | **0.827** | **0.137** | **36.23** |

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support (CPU execution supported but slow)
- Nvidia GPU with ~17 GB VRAM (RTX 5060 or better recommended)

### Conda Environment Setup

```bash
# Create a new conda environment
conda create -n solarflow python=3.9 -y
conda activate solarflow

# Install PyTorch with CUDA (adjust cuda version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt
```

### Pip-Only Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch` >= 2.0
- `numpy`, `pandas` - Numerical computing
- `xarray`, `h5py`, `cfgrib`, `eccodes` - Geospatial data I/O
- `Pillow`, `scikit-image` - Image processing
- `tqdm` - Progress bars

---

## Data

### Dataset Overview

| Dataset | Resolution | Channels | Role |
|---------|-----------|----------|------|
| **ERA5** | ~0.25° (~25 km) | 5 radiative | Coarse physical baseline |
| **SolarCube** | High-res (~1 km) | 5 auxiliary + SSR target | Cloud context + ground truth |

**ERA5 Radiative Channels (5):**
- `era_ssrd`: Surface solar radiation downwards
- `era_ssr`: Surface net solar radiation
- `era_ssrdc`: Clear-sky surface solar radiation downwards
- `era_fdir`: Total sky direct solar radiation
- `era_cdir`: Clear-sky direct solar radiation

**SolarCube Auxiliary Channels (5):**
- `vis047`, `vis086`: Visible satellite reflectance
- `ir133`: Infrared brightness temperature
- `sza`: Solar zenith angle
- `cm`: Cloud mask

**Target:**
- `solarcube_ssr_hourly`: Hourly mean high-resolution SSR field

### Dataset Split

Day-blocked train/validation/test split with 70/10/20 ratio:

| Split | Samples | Fraction |
|-------|---------|----------|
| Train | 32,159 | 70% |
| Validation | 4,519 | 10% |
| Test | 8,994 | 20% |

**Tiles used:** 1-10, 12, 14 (12 global stations passing quality-control threshold)
**Split strategy:** Whole UTC-day blocks within each tile/month to prevent temporal leakage

### Directory Structure

After preprocessing, data should be organized as:

```
data/
├── raw/
│   ├── ERA5/
│   │   ├── era5_YYYYMM_tile_XX.grib
│   │   └── ...
│   └── SolarCube/
│       ├── solarcube_YYYYMM_tile_XX.nc
│       └── ...
└── processed/
    └── tile_XX/
        └── month_YY/
            └── chunks/
                ├── sample_0000.npz
                ├── sample_0001.npz
                └── ...
```

### Download & Preprocessing

1. **Download raw data** (ERA5 and SolarCube files) and place in `data/raw/`

2. **Check inputs:**
   ```bash
   python preprocess.py check
   ```

3. **Preprocess raw files:**
   ```bash
   python preprocess.py preprocess --tiles 1-10,12,14 --months 1-12
   ```

4. **Create day-block splits:**
   ```bash
   python preprocess.py splits --split-tiles 1-10,12,14 --split-months 1-12 --seed 42
   ```

This generates `splits/train_index.csv`, `splits/val_index.csv`, and `splits/test_index.csv`.

---

## Training

### Quick Start: End-to-End SolarFlowRefiner (Recommended)

Train the refinement-aware model jointly from scratch:

```bash
python train.py --model solarflowrefiner \
  -- --train-index splits/train_index.csv \
     --val-index splits/val_index.csv \
     --out-dir runs/SolarFlowRefiner \
     --epochs 100
```

**Training details:**
- Batch size: 1
- Learning rates: 2×10⁻⁶ (FlowMatch generator), 2×10⁻⁵ (refiner)
- Optimizer: AdamW with weight decay 10⁻⁴
- EMA decay: 0.9999
- Refinement levels: K=8
- Noise schedule: σ ∈ [0.35, 0.01]
- Hardware: Nvidia RTX 5060 (~17 GB VRAM)

### Two-Stage Workflow (Baseline)

#### Stage 1: Train Standalone FlowMatch Generator

```bash
python train.py --model flowmatch \
  -- --train-index splits/train_index.csv \
     --val-index splits/val_index.csv \
     --out-dir runs/FlowMatch \
     --epochs 100 \
     --batch-size 4 \
     --lr 2e-5
```

#### Stage 2a: Post-Hoc ODE Refinement

```bash
# Precompute base predictions
python run_experiment.py --model flowrefiner-ode --stage precompute \
  -- --index splits/train_index.csv \
     --run-dir runs/FlowMatch \
     --out runs/base_cache/train_base.npy

python run_experiment.py --model flowrefiner-ode --stage precompute \
  -- --index splits/val_index.csv \
     --run-dir runs/FlowMatch \
     --out runs/base_cache/val_base.npy

# Train ODE refiner (frozen generator)
python train.py --model flowrefiner-ode \
  -- --train-index splits/train_index.csv \
     --val-index splits/val_index.csv \
     --train-base-cache runs/base_cache/train_base.npy \
     --val-base-cache runs/base_cache/val_base.npy \
     --base-run-dir runs/FlowMatch \
     --out-dir runs/FlowRefiner-ODE \
     --epochs 50
```

#### Stage 2b: Post-Hoc PDE Refinement

```bash
# Precompute base predictions (if not already done)
python run_experiment.py --model flowrefiner-pde --stage precompute \
  -- --index splits/train_index.csv \
     --run-dir runs/FlowMatch \
     --out runs/base_cache/train_base.npy

python run_experiment.py --model flowrefiner-pde --stage precompute \
  -- --index splits/val_index.csv \
     --run-dir runs/FlowMatch \
     --out runs/base_cache/val_base.npy

# Train PDE refiner (frozen generator)
python train.py --model flowrefiner-pde \
  -- --train-index splits/train_index.csv \
     --val-index splits/val_index.csv \
     --train-base-cache runs/base_cache/train_base.npy \
     --val-base-cache runs/base_cache/val_base.npy \
     --base-run-dir runs/FlowMatch \
     --out-dir runs/FlowRefiner-PDE \
     --epochs 50
```

### Evaluation

```bash
python evaluate.py --model solarflowrefiner \
  -- --test-index splits/test_index.csv \
     --run-dir runs/SolarFlowRefiner
```

Metrics computed: MAE, RMSE, SSIM, LPIPS, FID

---

## Method Summary

SolarFlowRefiner introduces **refinement-aware flow matching** for solar downscaling:

### 1. Residual Formulation

Rather than predicting SSR directly, we model the normalized residual correction:

```
r = y - b                     (residual)
r̃ = (r - μ_r) / σ_r          (normalized residual)
ŷ = b + σ_r · r̂̃ + μ_r        (reconstruction)
```

where `b` is the upsampled ERA5 SSR baseline, `y` is the SolarCube target, and `μ_r, σ_r` are training-set residual statistics.

### 2. Conditional FlowMatch Predictor

A U-Net generator learns a rectified flow between Gaussian noise `z_0 ~ N(0,I)` and the normalized target residual `z_1 = r̃`:

```
z_t = (1-t)z_0 + t·z_1                      (linear path)
L_FM = E[ || f_θ(z_t, x, t) - (z_1 - z_0) ||_1 ]   (velocity objective)
```

At inference, the ODE `dz/dt = f_θ(z, x, t)` is integrated from `t=0` to `t=1` using an 8-step Euler solver to produce the base residual `r̃_b`.

### 3. Prediction-Conditioned Refinement Path

Instead of perturbing the ground-truth target, we construct refinement states from the **current generator prediction**:

```
c_k = (1 - α_k)·r̃_b + α_k·r̃              (base-to-target path)
c̄_k = c_k + σ_k·ε, ε ~ N(0,I)            (noisy state)
```

where `α_k = k/(K-1)` and `σ_k` decays exponentially from 0.35 to 0.01 over K=8 levels. This exposes the refiner to the **structured errors** produced by the generator.

### 4. PDE-Style Multilevel Refiner

The refiner `g_φ` predicts the clean target residual from noisy intermediate states:

```
r̂_k = g_φ(c̄_k, x, r̃_b, k)

L_ref = || r̂_k - r̃ ||_1 + λ_mse·|| r̂_k - r̃ ||_2² + λ_∇·|| ∇r̂_k - ∇r̃ ||_1
```

At inference, refinement proceeds iteratively:

```
u_0 = r̃_b                                 (initialize from base)
u_{k+1} = u_k + η·(r̂_k - u_k)            (correction update)
```

### 5. Refinement-Aware Joint Optimization

The key distinction: SolarFlowRefiner runs the differentiable FlowMatch sampler **inside the training loop**:

```
L_joint = λ_FM·L_FM + λ_ref·L_ref
```

Because `r̃_b = S_θ(z_0, x)` appears in both the refinement path and refiner conditioning, gradients flow back through the sampler:

```
∇_θ L_ref ≠ 0
```

This allows the generator and refiner to **co-adapt**: the generator learns to produce residuals that are easy to refine, while the refiner trains on prediction-conditioned states.

### Model Variants

| Model | Generator | Refiner | Coupling |
|-------|-----------|---------|----------|
| **FlowMatch** | FlowMatch | None | N/A |
| **FlowRefiner-ODE** | FlowMatch (frozen) | ODE correction | Post-hoc |
| **FlowRefiner-PDE** | FlowMatch (frozen) | PDE denoising | Post-hoc |
| **SolarFlowRefiner** | FlowMatch (trainable) | PDE denoising | **End-to-end** |

---

## Repository Structure

```
SolarFlowRefiner/
├── Model/
│   ├── FlowMatch/              # Standalone flow matching generator
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── model.py
│   ├── FlowRefiner-ODE/        # Post-hoc ODE-style refiner
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── model.py
│   ├── FlowRefiner-PDE/        # Post-hoc PDE-style refiner
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── model.py
│   ├── SolarFlowRefiner/       # End-to-end refinement-aware model
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── model.py
│   └── unet.py                 # Shared U-Net backbone
├── Dataset/
│   └── preprocessing/          # Data pipeline scripts
│       ├── build_dataset.py
│       ├── dataset_builder.py
│       ├── check_inputs.py
│       ├── create_splits.py
│       └── run_pipeline.py
├── utils/
│   ├── core.py                 # Data loading, statistics, path handling
│   └── metrics.py              # SSIM evaluation
├── configs/                    # Model configuration files (JSON)
├── scripts/                    # Example training/evaluation commands
├── splits/                     # Train/val/test manifests (CSV)
│   ├── train_index.csv
│   ├── val_index.csv
│   └── test_index.csv
├── data/                       # Dataset directory
│   ├── raw/                    # Raw ERA5 and SolarCube files
│   └── processed/              # Preprocessed NPZ samples
├── runs/                       # Training run outputs
├── train.py                    # Top-level training router
├── evaluate.py                 # Top-level evaluation router
├── run_experiment.py           # Model/stage dispatcher
├── preprocess.py               # Data preprocessing entry point
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{anonymous2027solarflowrefiner,
  title={SolarFlowRefiner: Refinement-Aware Flow Matching for Surface Solar Radiation Downscaling},
  author={Anonymous Authors},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2027}
}
```

**Related Work:**

```bibtex
@article{flowmatch,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matthew},
  journal={arXiv preprint arXiv:2210.02747},
  year={2022}
}

@inproceedings{pderefiner,
  title={PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers},
  author={Lippe, Phillip and Veeling, Bastiaan S. and Perdikaris, Paris and Turner, Richard E. and Brandstetter, Johannes},
  booktitle={NeurIPS},
  year={2023}
}

@article{flowrefiner,
  title={FlowRefiner: Refining Autoregressive 3D Turbulent Flow Prediction through Iterative Flow Matching},
  author={Dai, Longxiang and others},
  journal={arXiv preprint arXiv:2604.17149},
  year={2026}
}

@article{era5,
  title={The ERA5 global reanalysis},
  author={Hersbach, Hans and others},
  journal={Quarterly Journal of the Royal Meteorological Society},
  volume={146},
  pages={1999--2049},
  year={2020}
}

@article{solarcube,
  title={SolarCube: A high-resolution satellite-derived surface solar radiation dataset},
  author={others},
  journal={TBD},
  year={TBD}
}
```

---

## Acknowledgements

- **ERA5 Data**: Copernicus Climate Change Service (C3S)
- **SolarCube Dataset**: BSRN/PANGAEA station network
- **FlowRefiner**: Inspiration for refinement-aware generation framework
- **PyTorch Team**: Deep learning framework

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contact

For questions or issues, please open an issue on the repository or contact the authors.
