import mlflow


import matplotlib

matplotlib.use(
    "Agg"
)  # this is necessary because non-interactive matplotlib does not work

import matplotlib.pyplot as plt


import torch
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)


class MAEevaluatorIN:
    def __init__(
        self,
        dataset_name,
        imagenet_default_mean_and_std=True,
        fixed_post_norm_shift=1.860,
    ):
        self.dataset_name = dataset_name
        self.batch = None
        self.output = None
        self.fixed_post_norm_shift = float(fixed_post_norm_shift or 0.0)
        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = IMAGENET_INCEPTION_MEAN
            std = IMAGENET_INCEPTION_STD
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def evaluate_batch(self, model_output, batch):
        self.prediction, self.batch = model_output["predictions"], batch
        self.output = {
            "preds": model_output["predictions"],
            "mask": model_output["mask"],
            "masked": model_output["masked_data"],
            "original": model_output["input"],
            "feats": model_output["feats"],
        }

    def to_img(self, x, robust=False):
        x = x.detach().cpu()

        x = (x - self.fixed_post_norm_shift) * self.std + self.mean
        if robust:
            flat = x.flatten()
            lo = torch.quantile(flat, 0.01)
            hi = torch.quantile(flat, 0.99)
            x = (x - lo) / (hi - lo + 1e-6)
        x = x.clamp(0, 1)
        x = x.permute(1, 2, 0)  # HWC
        return x

    def log_epoch(self, epoch):
        if self.output is None:
            return
        preds = self.output["preds"]
        mask = self.output["mask"]
        masked = self.output["masked"]
        originals = self.output["original"]
        feats = self.output["feats"]
        num_rows = 10

        # MAE-style visualization: keep visible regions from input,
        # fill only masked regions with model predictions.
        recon = originals * mask + preds * (1 - mask)

        for i in range(num_rows):
            fig, axs = plt.subplots(1, 3, squeeze=False, figsize=(9, 3))
            fig.suptitle(f"Image {i+1}", y=1.02)
            axs[0, 0].imshow(self.to_img(originals[i]))
            axs[0, 0].set_title(f"Original")
            axs[0, 1].imshow(self.to_img(masked[i]))
            axs[0, 1].set_title(f"Masked")
            axs[0, 2].imshow(self.to_img(recon[i], robust=True))
            axs[0, 2].set_title(f"Reconstruction")
            for ax in axs[0]:
                ax.set_axis_off()

            plt.tight_layout()
            mlflow.log_figure(
                fig,
                f"{self.dataset_name}/predictions/image_{i}/epoch_{epoch}.png",
            )
            plt.close(fig)
