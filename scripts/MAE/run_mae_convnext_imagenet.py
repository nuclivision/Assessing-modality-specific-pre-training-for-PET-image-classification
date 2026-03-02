print("Loading MAE preprocessor...")
print("Importing necessary modules...")

from nucli_train.models.builders import build_model
from nucli_train.training import Trainer
import mlflow
import sys
from pathlib import Path
import yaml
import argparse

ROOT = Path(__file__).resolve().parents[2]  # repo root above scripts/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.imagenet_subset import build_imagenet_data
from src.val import evaluator_MAE_imagenet
import src.models.MAE_IN
import src.nets.convnext
from src.models.factory import build_local_model

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

parser.add_argument(
    "--experiment-name",
    type=str,
    default="MAE ConvNeXt",
    help="experiment name",
)
args = parser.parse_args()
print("Importing necessary modules completed.")

model_cfg = args.model_cfg
data_cfg = args.data_cfg
run_name = args.run_name
experiment_name = args.experiment_name

with open(model_cfg, "r") as f:
    model_cfg_dict = yaml.safe_load(f)

print("Build model")
model_cfg_local = model_cfg_dict["model"] if "model" in model_cfg_dict else model_cfg_dict
try:
    model = build_local_model(model_cfg_local)
except ValueError:
    model = build_model(model_cfg_dict)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model params: {total_params:,}")

print("Build data")
train_data, val_loaders = build_imagenet_data(data_cfg)

print("Build trainer")
if mlflow.active_run():
    mlflow.end_run()

trainer = Trainer(
    model,
    train_data=train_data,
    val_loaders=val_loaders,
    run_name=run_name,
    experiment_name=experiment_name,
    save_interval=10,
    model_cfg=model_cfg,
    data_cfg=data_cfg,
)

print("Run trainer")
trainer.run(args.epochs)
