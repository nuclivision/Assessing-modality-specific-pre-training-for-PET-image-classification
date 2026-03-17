# Assessing Modality-Specific Pre-training for PET Image Classification

Codebase for the paper:

**When Does Modality-Specific Pre-training Help? A Controlled Study for PET Image Classification**
Anonymized Authors

## Overview

This repository implements a controlled study of self-supervised pre-training for FDG-PET classification with:

1. **MAE pre-training using a ConvNeXtV2 architecture**
2. **Downstream patient-level classification**

The core question is whether modality-specific PET pre-training improves performance over size-matched natural-image pre-training (ImageNet) when all other factors are kept fixed.

## Key Findings (Paper Summary)

Under identical pre-training budgets and matched dataset sizes:

- **Full fine-tuning (FT):** ImageNet pre-training is a very strong baseline and outperforms PET-only pre-training.
- **Two-phase pre-training:** IN1+PET and IN1+IN2 both improve over one-phase baselines in FT and reach similar top AUROC.
- **Linear probing (LP):** IN1+PET performs best, suggesting PET adaptation improves representation quality when the encoder is frozen.

Reported AUROC summary (from manuscript):

| Pre-training | #Cases | LP LOO | LP All | FT LOO | FT All |
|---|---:|---:|---:|---:|---:|
| PET | 14,000 | 0.59 [0.50,0.68] | 0.65 [0.56,0.74] | 0.76 [0.69,0.84] | 0.90 [0.85,0.94] |
| IN1 | 14,000 | 0.58 [0.49,0.67] | 0.72 [0.64,0.80] | 0.89 [0.84,0.94] | 0.94 [0.91,0.98] |
| IN1 + PET | 28,000 | 0.66 [0.57,0.75] | 0.74 [0.66,0.81] | 0.90 [0.84,0.95] | 0.96 [0.93,0.99] |
| IN1 + IN2 | 28,000 | 0.56 [0.47,0.66] | 0.67 [0.58,0.75] | 0.88 [0.82,0.93] | 0.96 [0.93,0.98] |


## Installation

Run `pip install -r requirements.txt`. Use a Python environment with Python 3.10 or newer. Our setup uses Python 3.11, but any version ≥ 3.10 is likely to work.

No separate package installation is required. The provided scripts are designed to run directly from the cloned repository and handle access to the `src/` code at runtime.


## Repository Structure

```
configs/
  data/                   # Data configs for PET MIPs and ImageNet subsets
  pretrain/               # Model configs for MAE pre-training
    MIP_pretraining/
    IN_pretraining/
    2phase_pretraining/
  classification/         # Setup config and transforms for downstream task
scripts/
  create_mips/            # Preprocessing: 3D NIfTI → MIP NIfTI
  pretrain/               # MAE pre-training and inference scripts
  classification/         # Downstream classification training scripts
src/
  data/                   # MIPDataset and ImageNet subset data loading
  models/                 # MAE and classifier model wrappers
  nets/                   # ConvNeXt backbone, sparse convolution blocks
  val/                    # MAE reconstruction evaluators
  inference/              # Inference utilities
```


## Data Format Expectations

### PET data (for MAE pre-training and downstream classification)

The code expects a center-wise folder structure:

```
<root>/
  <center_name>/
    fdg/
      pet/
        4_MIPs/
          <PatientID>.nii.gz
```

Each `<PatientID>.nii.gz` is a 3D NIfTI of shape `(n_angles, H, W)`, where `n_angles` is the number of MIP projections (default: 4). These files are produced by `scripts/create_mips/create_cropped_mips.py` (see Step 0 below).

### ImageNet subset data

Expected standard `ImageFolder` layout:

```
<imagenet_root>/
  train/
    <class_1>/*.jpg
    <class_2>/*.jpg
    ...
  val/
    <class_1>/*.jpg
    ...
```

## Experimental Design (Paper-Aligned)

- ConvNeXtV2-tiny MAE encoder (`depths=[3,3,9,3]`, `dims=[96,192,384,768]`)
- Mask ratio: `0.7`
- Patch size: `32`
- One-phase pre-training: 1,600 epochs
- Two-phase pre-training: 800 + 800 epochs
- Matched dataset sizes to isolate modality effects

---

## Step 0: Create MIP Images

PET volumes must first be projected into 2D Maximum Intensity Projections (MIPs). This step converts raw 3D NIfTI scans to multi-angle 2D MIP stacks.

The script expects input files at `<root_dir>/<tracer>/<scan>.nii.gz` and writes output to `<root_dir>/<tracer>/pet/<n>_MIPs/`.

```bash
python scripts/create_mips/create_cropped_mips.py \
  --root-dir /path/to/raw_pet_data \
  --n-angles 4 \
  --target-res 1.5 \
  --target-hw 480
```

Arguments:
- `--n-angles`: number of MIP projections evenly spaced over [0°, 180°) (default: 4)
- `--target-res`: isotropic voxel spacing in mm before projection (default: 1.5)
- `--target-hw`: height and width of output MIP images in pixels (default: 480)

---

## Step 1: MAE Pre-training

### 1.1 PET MAE pre-training

Edit `configs/data/mip_data.yaml` to set your data root:

```yaml
data_root: "/path/to/pet_root"
```

Run:

```bash
python scripts/pretrain/run_mae_convnext.py \
  --model-cfg configs/pretrain/MIP_pretraining/convnextMAE_T.yaml \
  --data-cfg configs/data/mip_data.yaml \
  --epochs 1600
```

### 1.2 ImageNet MAE pre-training (IN1)

Edit `configs/data/imagenet_subset.yaml` to set your data root:

```yaml
data_root: "/path/to/imagenet_root"
```

Run:

```bash
python scripts/pretrain/run_mae_convnext_imagenet.py \
  --model-cfg configs/pretrain/IN_pretraining/convnextMAE_T.yaml \
  --data-cfg configs/data/imagenet_subset.yaml \
  --run-name MAE_IN1 \
  --experiment-name "MAE ConvNeXt" \
  --epochs 1600
```

### 1.3 Two-phase pre-training (IN1 → PET)

Phase 1 (ImageNet):

```bash
python scripts/pretrain/run_mae_convnext_imagenet.py \
  --model-cfg configs/pretrain/IN_pretraining/convnextMAE_T.yaml \
  --data-cfg configs/data/imagenet_subset.yaml \
  --run-name MAE_IN1_phase1 \
  --epochs 800
```

Phase 2 (PET) — set the phase-1 checkpoint path in `configs/pretrain/2phase_pretraining/MIPs_convnextMAE_T.yaml`:

```yaml
model:
  args:
    network:
      args:
        prepretrained_ckpt: "/path/to/phase1_IN1_checkpoint.pth"
```

Then run:

```bash
python scripts/pretrain/run_mae_convnext.py \
  --model-cfg configs/pretrain/2phase_pretraining/MIPs_convnextMAE_T.yaml \
  --data-cfg configs/data/mip_data.yaml \
  --epochs 800
```

### 1.4 Two-phase pre-training (IN1 → IN2)

Use the same two-phase setup with:
- Phase 2 model config: `configs/pretrain/2phase_pretraining/IN_convnextMAE_T.yaml`
- A different ImageNet subset for IN2 (different `subset_seed` or `subset_class_seed` in `configs/data/imagenet_subset.yaml`)

### 1.5 Visualize MAE reconstructions

```bash
python scripts/pretrain/run_inf_mae.py \
  --model-cfg configs/pretrain/MIP_pretraining/convnextMAE_T.yaml \
  --ckpt-path /path/to/mae_checkpoint.pth \
  --mip-nii /path/to/sample_4mips.nii.gz \
  --out-path /path/to/output_triptych.png
```

This produces a side-by-side image per MIP: original — masked input — reconstruction.

---

## Step 2: Downstream Classification

Patient-level binary classification (normal vs. abnormal) using 4 MIPs with attention pooling over projections and a classification head on top of the frozen or partially fine-tuned encoder.

### 2.1 Configure the setup

Edit `configs/classification/setup_clf.yaml`:

```yaml
data:
  path_to_dataset: "/path/to/downstream_pet_root"
  nb_MIPs: 4

model:
  model_args:
    pretrain_source: "mae"        # "mae" or "timm"
    backbone: "/path/to/mae_checkpoint.pth"   # checkpoint path (mae) or timm model name (timm)
    linearprobe: true             # true = linear probe, false = fine-tune
    block_training_budget: 1      # number of ConvNeXt blocks to unfreeze from the end (ignored if linearprobe=true)
```

For timm initialization:

```yaml
model:
  model_args:
    pretrain_source: "timm"
    backbone: "convnextv2_tiny.fcmae_ft_in1k"
```

### 2.2 Run training

Log to MLflow:

```bash
python scripts/classification/train_clf.py \
  --setup-config configs/classification/setup_clf.yaml \
  --epochs 40 \
  --run-name IN1_PET_LP \
  --experiment_name "PET classification"
```

Log to Weights & Biases:

```bash
python scripts/classification/train_clf_WANDB.py \
  --setup-config configs/classification/setup_clf.yaml \
  --epochs 40 \
  --run-name IN1_PET_LP
```

Run a single cross-validation fold:

```bash
python scripts/classification/train_clf.py \
  --setup-config configs/classification/setup_clf.yaml \
  --epochs 40 \
  --cv-fold 1
```

### 2.3 Inference on a test set

Use `src/inference/inference.py` (`run_inference`) with:
- setup config path
- classifier checkpoint path
- CSV containing `PatientID`, `Label`, `image`

The output includes per-patient prediction and positive-class probability (`Prob_Pos`).

---

## Reproducing the Controlled Comparison

1. Train MAE models with matched budgets:
   - PET (14k)
   - IN1 (14k)
   - IN1+PET (28k across two phases)
   - IN1+IN2 (28k across two phases)
2. Train the downstream classifier for each initialization:
   - LP (`linearprobe: true`)
   - FT (`linearprobe: false`, tune `block_training_budget`)
3. Evaluate pooled and leave-one-center-out performance with AUROC and confidence intervals.

---

## Architecture Notes

### Sparse ConvNeXt encoder

The MAE encoder is a ConvNeXtV2-tiny modified to use **sparse convolutions** during pre-training. This follows the [SparK](https://github.com/keyu-tian/SparK) approach: masked patches are zeroed out and the active (unmasked) region is tracked via a binary mask. Convolutions are applied only over active locations, so masked regions do not contribute to feature statistics.

### Reweighted sparse convolution (custom modification)

Standard sparse convolution computes a weighted sum over a convolution window, but when the window overlaps the mask boundary, some kernel positions land on zeroed-out (inactive) pixels. This dilutes the output compared to a convolution that only sees active pixels — the effective normalization of the kernel is smaller near boundaries.

We introduce `SparseConv2dReweighted` (`src/nets/sparse_transform.py`) to correct for this. This is **not part of the original SparK implementation** and was added to prevent masked (zero-padded) regions from biasing the convolution output at active-region borders. The correction works as follows:

1. **Apply the convolution** on the masked input (inactive pixels are zero).
2. **Compute how much of each kernel was actually active** for every output position, by convolving the binary input mask with the absolute kernel weights. This gives the effective kernel mass `mask_sum` at each position.
3. **Rescale the output** so that the contribution is normalized relative to the full kernel mass `full_sum`:

   ```
   output_normalized = output_no_bias / (mask_sum + ε) × full_sum
   ```

4. **Zero out inactive output positions** using the output mask, so sparse regions remain inactive in the next layer.

The effect is that a border convolution window with only half its support over active pixels will have its output scaled up to match what a fully-active window would have produced — preventing the artificial suppression of activations at the active-region boundary caused by zero-padded masked pixels.

---

## Notes and Practical Tips

- **Intensity handling:** PET inputs use a log-intensity transform (`intensitylog: true` in the pretrain config). This is applied inside the model and does not require preprocessing. ImageNet pre-training uses standard normalization (`intensitylog: false`).
- **Channel mismatch:** PET inputs are single-channel. When using a timm backbone expecting 3-channel input, channel repetition is handled automatically in `ConvNeXtClassifier.forward`.
- **Two-phase checkpoint loading:** Phase-2 training loads the phase-1 checkpoint via `prepretrained_ckpt` with non-strict weight loading, so only encoder weights are transferred.
- **Label parsing:** Labels are derived from patient IDs by default (suffix `_N` → 0, `_AN` → 1). If your naming convention differs, configure `label_from` and `label_map` under `data:` in `setup_clf.yaml`.
- **Stale dataframe cache:** Patient CSVs are cached under `<data_root>/dataframes/`. If you change the dataset or filters, delete the cached CSV to force regeneration.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{anonymous2026modalityspecificpet,
  title={When Does Modality-Specific Pre-training Help? A Controlled Study for PET Image Classification},
  author={Anonymized Authors},
  year={2026}
}
```

## License

This repository is released under the license in [LICENSE](LICENSE).
