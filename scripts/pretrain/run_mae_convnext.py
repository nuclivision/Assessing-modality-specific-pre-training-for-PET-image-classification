import mlflow
import sys
from pathlib import Path
import yaml
import argparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.MIPdataset import build_mip_data
from src.models.factory import build_model
from src.trainer import Trainer


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="../../configs/models/MIP_pretraining/convnextMAE_T.yaml",
        help="Path to model config YAML",
    )

    parser.add_argument(
        "--data-cfg",
        type=str,
        default="../../configs/data/mip_data.yaml",
        help="Path to data config YAML",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1600,
        help="nb of epochs",
    )

    parser.add_argument("--save-interval", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_cfg = args.model_cfg
    data_cfg = args.data_cfg

    with open(model_cfg, "r") as f:
        model_cfg_dict = yaml.safe_load(f)

    model = build_model(model_cfg_dict)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model params: {total_params:,}")

    print("Build data")
    train_data, val_loaders = build_mip_data(data_cfg)

    print("Build trainer")
    if mlflow.active_run():
        mlflow.end_run()

    trainer = Trainer(
        model,
        train_data=train_data,
        val_loaders=val_loaders,
        run_name="convnextMAE_PPT",
        experiment_name="MAE convnext",
        save_interval=100,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
    )

    print("Run trainer")
    trainer.run(args.epochs)
