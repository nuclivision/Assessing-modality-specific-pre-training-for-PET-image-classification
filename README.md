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

## Repository Structure

```text
configs/
  data/
    mip_data.yaml
    imagenet_subset.yaml
  models/
    MIP_pretraining/convnextMAE_T.yaml
    IN_pretraining/convnextMAE_T.yaml
    2phase_pretraining/
      MIPs_convnextMAE_T.yaml
      IN_convnextMAE_T.yaml
  tasks/
    setup_MAEclf.yaml
    MIP-transforms.yaml

scripts/MAE/
  run_mae_convnext.py              # PET MAE pre-training
  run_mae_convnext_imagenet.py     # ImageNet MAE pre-training
  run_inf_mae.py                   # MAE reconstruction visualization

tasks/scripts/
  train_clf.py                     # downstream classifier training (MLflow)
  train_clf_WANDB.py               # downstream classifier training (W&B)
  run_clf.py                       # wrapper for train_clf_WANDB

src/
  data/
  models/
  nets/
  inference/
```

## Environment and Dependencies

### Python

- Recommended: Python 3.10+

### Required packages

The code imports at least:

- `torch`, `torchvision`
- `monai`
- `timm`
- `mlflow`
- `wandb` (for the W&B training script)
- `numpy`, `pandas`, `scikit-learn`
- `nibabel`
- `pyyaml`
- `matplotlib`
- `nucli_train` (required by training entrypoints)

Example installation (adapt to your CUDA/PyTorch setup):

```bash
pip install torch torchvision monai timm mlflow wandb scikit-learn nibabel pandas pyyaml matplotlib
```

`nucli_train` must also be installed/available in your environment.

## Data Format Expectations

### PET data (for MAE pre-training and downstream classification)

The code expects center-wise folder structure:

```text
<root>/
  <center_name>/
    fdg/
      pet/
        4_MIPs/
          <PatientID>.nii.gz
```

Alternative locations are partially supported (e.g., directly under `pet/`), but `4_MIPs/` is the default used in the data loaders.

### ImageNet subset data

Expected standard `ImageFolder` layout:

```text
<imagenet_root>/
  train/
    <class_1>/*.jpg
    <class_2>/*.jpg
    ...
  val/
    <class_1>/*.jpg
    <class_2>/*.jpg
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

## 1. MAE Pre-training Using a ConvNeXtV2 Architecture

This section corresponds to SSL pre-training before the downstream classifier.

### 1.1 PET MAE pre-training

Use:
- Script: `scripts/MAE/run_mae_convnext.py`
- Model config: `configs/models/MIP_pretraining/convnextMAE_T.yaml`
- Data config: `configs/data/mip_data.yaml`

Edit `configs/data/mip_data.yaml` first:

```yaml
data_root: "/path/to/pet_root"
nb_MIPs: 4
patch_size: [224, 224]
transforms: ["rotate", "flip"]
```

Run:

```bash
python scripts/MAE/run_mae_convnext.py \
  --model-cfg configs/models/MIP_pretraining/convnextMAE_T.yaml \
  --data-cfg configs/data/mip_data.yaml \
  --epochs 1600
```

### 1.2 ImageNet MAE pre-training (IN1)

Use:
- Script: `scripts/MAE/run_mae_convnext_imagenet.py`
- Model config: `configs/models/IN_pretraining/convnextMAE_T.yaml`
- Data config: `configs/data/imagenet_subset.yaml`

Edit `configs/data/imagenet_subset.yaml` first:

```yaml
data_root: "/path/to/imagenet_root"
train_subset_size: 14000
subset_num_classes: 100
subset_per_class: true
```

Run:

```bash
python scripts/MAE/run_mae_convnext_imagenet.py \
  --model-cfg configs/models/IN_pretraining/convnextMAE_T.yaml \
  --data-cfg configs/data/imagenet_subset.yaml \
  --run-name MAE_IN1 \
  --experiment-name "MAE ConvNeXt" \
  --epochs 1600
```

### 1.3 Two-phase pre-training (IN1 -> PET)

Phase 1 (ImageNet IN1):

```bash
python scripts/MAE/run_mae_convnext_imagenet.py \
  --model-cfg configs/models/IN_pretraining/convnextMAE_T.yaml \
  --data-cfg configs/data/imagenet_subset.yaml \
  --run-name MAE_IN1_phase1 \
  --epochs 800
```

Phase 2 (PET), initialize from phase-1 checkpoint:

1. Set checkpoint path in `configs/models/2phase_pretraining/MIPs_convnextMAE_T.yaml`:

```yaml
model:
  args:
    network:
      args:
        prepretrained_ckpt: "/path/to/phase1_IN1_checkpoint.pth"
```

2. Run phase 2:

```bash
python scripts/MAE/run_mae_convnext.py \
  --model-cfg configs/models/2phase_pretraining/MIPs_convnextMAE_T.yaml \
  --data-cfg configs/data/mip_data.yaml \
  --epochs 800
```

### 1.4 Two-phase pre-training (IN1 -> IN2)

Use the same two-phase idea with:

- Phase-2 model config: `configs/models/2phase_pretraining/IN_convnextMAE_T.yaml`
- A different ImageNet subset definition for IN2 (e.g., different subset seed and/or class seed in `configs/data/imagenet_subset.yaml`)

### 1.5 Visualize MAE reconstructions

Use:
- Script: `scripts/MAE/run_inf_mae.py`

```bash
python scripts/MAE/run_inf_mae.py \
  --model-cfg configs/models/MIP_pretraining/convnextMAE_T.yaml \
  --ckpt-path /path/to/mae_checkpoint.pth \
  --mip-nii /path/to/sample_4mips.nii.gz \
  --out-path /path/to/output_triptych.png
```

This produces a triptych per MIP: original, masked input, and reconstruction.

---

## 2. Downstream Classification

This section corresponds to patient-level binary classification (normal vs abnormal) using 4 MIPs + attention pooling + classifier head.

### 2.1 Model and training modes

Configured in `configs/tasks/setup_MAEclf.yaml`:

- `model.name: classifier_MAE`
- `model.model_args.pretrain_source`:
  - `"mae"`: load your own MAE checkpoint
  - `"timm"`: use official timm pretrained ConvNeXtV2
- `model.model_args.linearprobe`:
  - `true`: linear probing
  - `false`: full fine-tuning (with classifier head + unfreezing policy)
- `model.model_args.block_training_budget`:
  - number of ConvNeXt blocks to unfreeze from the end of the encoder

For MAE initialization, set:

```yaml
model:
  model_args:
    pretrain_source: "mae"
    backbone: "/path/to/mae_checkpoint.pth"
```

For timm initialization, set:

```yaml
model:
  model_args:
    pretrain_source: "timm"
    timm_backbone: "convnextv2_tiny.fcmae_ft_in1k"
```

### 2.2 Configure dataset paths

In `configs/tasks/setup_MAEclf.yaml`:

```yaml
data:
  path_to_dataset: "/path/to/downstream_pet_root"
  nb_MIPs: 4
```

Transforms are loaded from:

- `configs/tasks/MIP-transforms.yaml`

### 2.3 Run downstream training (MLflow)

```bash
python tasks/scripts/train_clf.py \
  --setup-config configs/tasks/setup_MAEclf.yaml \
  --epochs 300 \
  --run-name IN1_PET_FT \
  --experiment_name "PET classification"
```

Run one CV fold only:

```bash
python tasks/scripts/train_clf.py \
  --setup-config configs/tasks/setup_MAEclf.yaml \
  --epochs 300 \
  --cv-fold 1
```

### 2.4 Run downstream training (W&B)

```bash
python tasks/scripts/run_clf.py \
  --setup-config configs/tasks/setup_MAEclf.yaml \
  --epochs 300 \
  --run-name IN1_PET_FT \
  --experiment_name "PET classification"
```

### 2.5 Inference on a test CSV

Use `src/inference/inference.py` functions (`run_inference`) with:

- setup config path
- classifier checkpoint path
- CSV containing `PatientID`, `Label`, `image`

The output includes per-patient prediction and positive-class probability (`Prob_Pos`).

## Reproducing the Controlled Comparison

To reproduce the study logic:

1. Train MAE models with matched budgets:
   - PET (14k)
   - IN1 (14k)
   - IN1+PET (28k across two phases)
   - IN1+IN2 (28k across two phases)
2. Train downstream classifier for each initialization:
   - LP (`linearprobe: true`)
   - FT (`linearprobe: false`, tune unfreezing budget)
3. Evaluate pooled and leave-one-center-out performance with AUROC and confidence intervals.

## Notes and Practical Tips

- Intensity handling differs between PET and ImageNet pre-training (`intensitylog` flag in model configs).
- For PET inputs with single channel and RGB backbones, channel repetition is handled in model code.
- Checkpoint loading between phases uses `prepretrained_ckpt` and non-strict loading by design.
- If your volume naming or labels differ from defaults, update label parsing options in `train_clf.py` (`label_from`, `label_map` keys in setup data config).

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
