import mlflow
import sys
from pathlib import Path
import yaml
import argparse

ROOT = Path(__file__).resolve().parents[2]  # repo root above scripts/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.imagenet_subset import build_imagenet_data
from src.models.factory import build_model
from src.trainer import Trainer


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="../../configs/models/IN_pretraining/convnextMAE_T.yaml",
        help="Path to model config YAML",
    )

    parser.add_argument(
        "--data-cfg",
        type=str,
        default="../../configs/data/imagenet_subset.yaml",
        help="Path to data config YAML",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="MAE_IN_pre-training",
        help="runname",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1600,
        help="nb of epochs",
    )

    parser.add_argument("--save-interval", type=int, default=100)

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="MAE ConvNeXt",
        help="experiment name",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_cfg = args.model_cfg
    data_cfg = args.data_cfg
    run_name = args.run_name
    experiment_name = args.experiment_name

    with open(model_cfg, "r") as f:
        model_cfg_dict = yaml.safe_load(f)
    model = build_model(model_cfg_dict)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model params: {total_params:,}")

    train_data, val_loaders = build_imagenet_data(data_cfg)

    if mlflow.active_run():
        mlflow.end_run()

    trainer = Trainer(
        model,
        train_data=train_data,
        val_loaders=val_loaders,
        run_name=run_name,
        experiment_name=experiment_name,
        save_interval=args.save_interval,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
    )

    trainer.run(args.epochs)
