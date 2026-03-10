import mlflow

import numpy as np

import matplotlib.pyplot as plt
import torch


class MAEevaluator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.batch = None
        self.output = None
        self.intensitylog = False

    def evaluate_batch(self, model_output, batch):
        self.prediction, self.batch = model_output["predictions"], batch
        self.intensitylog = bool(model_output.get("intensitylog", False))
        self.output = {
            "preds": model_output["predictions"],
            "mask": model_output["mask"],
            "masked": model_output["masked_data"],
            "original": model_output["input"],
            "patient_id": model_output["patient_id"],
            "feats": model_output["feats"],
        }

    def _invert_preprocess(self, x):
        if self.intensitylog:
            return torch.expm1(x)
        return x

    def log_epoch(self, epoch):
        if self.output is None:
            return
        preds = self._invert_preprocess(self.output["preds"])
        mask = self.output["mask"]
        masked = self._invert_preprocess(self.output["masked"])
        originals = self._invert_preprocess(self.output["original"])
        recon = self.output["preds"] * mask + preds * (1 - mask)
        patient_id = self.output["patient_id"]
        feats = self.output["feats"]
        num_rows = min(preds.shape[0], len(patient_id), 5)

        for i in range(num_rows):
            sample_min = float(originals[i, 0].min().item())
            sample_max = float(originals[i, 0].max().item())
            fig, axs = plt.subplots(1, 3, squeeze=False, figsize=(9, 3))
            axs[0, 0].imshow(
                np.rot90(originals[i, 0].cpu().numpy(), k=1),
                cmap="gray",
                vmin=sample_min,
                vmax=sample_max,
            )
            axs[0, 0].set_title(f"Original {patient_id[i]}")
            axs[0, 1].imshow(
                np.rot90(masked[i, 0].cpu().numpy(), k=1),
                cmap="gray",
                vmin=sample_min,
                vmax=sample_max,
            )
            axs[0, 1].set_title(f"Masked {patient_id[i]}")
            axs[0, 2].imshow(
                np.rot90(recon[i, 0].cpu().numpy(), k=1),
                cmap="gray",
                vmin=sample_min,
                vmax=sample_max,
            )
            axs[0, 2].set_title(f"Reconstruction {patient_id[i]}")
            for ax in axs[0]:
                ax.set_axis_off()

            plt.tight_layout()
            mlflow.log_figure(
                fig,
                f"{self.dataset_name}/predictions/{patient_id[i]}/epoch_{epoch}.png",
            )
            plt.close(fig)
