import contextlib
import os
import random
from typing import Any
from os.path import join, normpath, sep

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Seeder:
    def __init__(self, base_seed):
        self.base_seed = base_seed

    def __call__(self, worker_id):
        worker_seed = self.base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_data=None,
        val_loaders={},
        run_name: str = None,
        experiment_name: str = None,
        use_amp: bool = False,
        tcompile: bool = False,
        save_interval: int = 200,
        resuming: dict[str, Any] = {"epoch": 0, "weights_path": None, "opt": False},
        save_opt: bool = True,
        model_cfg: dict | str = None,
        data_cfg: dict | str = None,
        trainer_cfg: dict | str = None,
        data_base_seed: int = 1,
    ):

        self.model = model
        if train_data is not None:
            self.train_dataset = train_data["dataset"]
            self.train_batch_size = train_data["batch_size"]
            self.train_workers = train_data["num_workers"]
        self.val_loaders = val_loaders
        self.use_amp = use_amp
        self.save_interval = save_interval
        self.compile = tcompile
        self.save_opt = save_opt
        self.data_base_seed = data_base_seed

        self.checkpoint_dir = None

        self.scaler = GradScaler() if self.use_amp else None

        self.model.cuda()

        self.starting_epoch = 1
        if run_name and experiment_name:

            self.optimizer = (
                self.model.get_optimizer()
            )  # list, same order as how losses get returned
            self.schedulers = (
                self.model.get_schedulers()
                if hasattr(self.model, "get_schedulers")
                else []
            )

            self.checkpoint_dir = join(
                os.environ.get("TRAIN_DIR", "./experiments"), experiment_name, run_name
            )

            if os.path.isdir(self.checkpoint_dir) and resuming["weights_path"] is None:
                if len(
                    os.listdir(self.checkpoint_dir)
                ) == 1 and "mlflow.yaml" in os.listdir(self.checkpoint_dir):
                    os.remove(join(self.checkpoint_dir, "mlflow.yaml"))
                if resuming["weights_path"] is None:
                    assert (
                        len(os.listdir(self.checkpoint_dir)) == 0
                    ), f"Checkpoint directory {self.checkpoint_dir} is not empty. Please remove it or choose a different run name."

            os.makedirs(self.checkpoint_dir, exist_ok=True)

            mlflow.set_tracking_uri(
                os.environ.get("MLFLOW_DIR", "./experiments/mlflow")
            )

            if resuming["weights_path"] is not None:
                self.model._load_checkpoint(
                    resuming["weights_path"],
                    resuming["epoch"],
                    resuming.get("opt", None),
                )
                if normpath(resuming["weights_path"]).split(sep)[-2:] == [
                    experiment_name,
                    run_name,
                ]:
                    self.starting_epoch = resuming["epoch"] + 1
                mlflow.set_experiment(experiment_name)
                r_id = yaml.safe_load(
                    open(join(resuming["weights_path"], "mlflow.yaml"))
                )["run_id"]
                if resuming.get("fresh_run", False):
                    mlflow.start_run(run_name=run_name)
                else:
                    mlflow.start_run(run_id=r_id)
            else:
                mlflow.set_experiment(experiment_name)
                mlflow.start_run(run_name=run_name)
                r_id = mlflow.active_run().info.run_id
                yaml.dump(
                    {"run_id": r_id},
                    open(join(self.checkpoint_dir, "mlflow.yaml"), "w"),
                )

            if isinstance(model_cfg, str):
                assert os.path.exists(
                    model_cfg
                ), f"Model config path {model_cfg} does not exist."
                assert model_cfg.endswith(".yaml") or model_cfg.endswith(
                    ".json"
                ), f"Model config path {model_cfg} is not a YAML or JSON file."
                model_cfg = yaml.safe_load(open(model_cfg, "r"))

            assert isinstance(
                model_cfg, dict
            ), f"The model_cfg must be a path or dict, got {model_cfg}"
            mlflow.log_dict(model_cfg, artifact_file="configs/model_cfg.yaml")

            if isinstance(data_cfg, str):
                assert os.path.exists(
                    data_cfg
                ), f"Data config path {data_cfg} does not exist."
                assert data_cfg.endswith(".yaml") or data_cfg.endswith(
                    ".json"
                ), f"Data config path {data_cfg} is not a YAML or JSON file."

                data_cfg = yaml.safe_load(open(data_cfg, "r"))

            assert isinstance(data_cfg, dict)
            mlflow.log_dict(data_cfg, artifact_file="configs/data_cfg.yaml")
            if trainer_cfg is not None:
                if isinstance(trainer_cfg, str):
                    assert os.path.exists(
                        trainer_cfg
                    ), f"Trainer config path {trainer_cfg} does not exist."
                    assert trainer_cfg.endswith(".yaml") or trainer_cfg.endswith(
                        ".json"
                    ), f"Trainer config path {trainer_cfg} is not a YAML or JSON file."

                    trainer_cfg = yaml.safe_load(open(trainer_cfg, "r"))

                assert isinstance(trainer_cfg, dict)
                mlflow.log_dict(trainer_cfg, artifact_file="configs/trainer_cfg.yaml")

            mlflow.log_param("batch_size", self.train_batch_size)
            for p, v in self.model.get_params().items():
                mlflow.log_param(p, v)

    def run(self, epochs):
        assert self.train_dataset is not None, "Train dataset must be provided."
        base_seed = self.data_base_seed
        for epoch in range(self.starting_epoch, epochs + 1):
            seed = base_seed + epoch
            seed_everything(seed)
            self.model.train()
            running_losses = {}

            generator = torch.Generator()
            generator.manual_seed(seed)
            seed_worker = Seeder(seed)
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                generator=generator,
                shuffle=True,
                num_workers=self.train_workers,
                worker_init_fn=seed_worker,
            )
            num_batches = len(train_loader)
            num_samples = len(self.train_dataset)
            bs_last_batch = (
                num_samples % self.train_batch_size
                if (num_samples % self.train_batch_size != 0)
                else self.train_batch_size
            )
            loader = tqdm(train_loader, desc=f"Epoch {epoch} / {epochs}", unit="batch")

            for b_idx, batch in enumerate(loader):
                if self.use_amp:
                    with autocast("cuda", dtype=torch.float16):
                        losses = self.model.train_step(batch)
                else:
                    losses = self.model.train_step(batch)

                loss_value = losses["value"]

                if self.use_amp:
                    self.scaler.scale(loss_value).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_value.backward()
                    self.optimizer.step()

                self.optimizer.zero_grad()
                bs_here = (
                    self.train_batch_size if b_idx != num_batches - 1 else bs_last_batch
                )
                for loss_name, val in losses.items():
                    if loss_name not in running_losses.keys():
                        running_losses[loss_name] = 0.0

                    running_losses[loss_name] += (val.item() * bs_here) / num_samples

            for loss_name, avg_loss in running_losses.items():
                mlflow.log_metric(f"train/{loss_name}", avg_loss, step=epoch)
            loader.close()

            self.validate(epoch)

            if epoch % self.save_interval == 0:
                nets_to_save = self.model.nets_to_save()
                for name, subnet in nets_to_save.items():
                    if isinstance(submodel, dict):
                        torch.save(
                            subnet,
                            os.path.join(
                                self.checkpoint_dir,
                                name + "_epoch_" + str(epoch) + ".pt",
                            ),
                        )
                    else:
                        torch.save(
                            subnet.state_dict(),
                            os.path.join(
                                self.checkpoint_dir,
                                name + "_epoch_" + str(epoch) + ".pt",
                            ),
                        )
                if self.save_opt:
                    self.model.save_opt(self.checkpoint_dir, str(epoch))

            for scheduler in self.schedulers:
                scheduler.step()

    def validate(self, epoch):
        self.model.eval()

        for dataset_name, loader_details in self.val_loaders.items():
            if epoch % loader_details["interval"] != 0:
                continue
            loader = loader_details["loader"]
            evaluators = loader_details["evaluators"]
            num_batches = len(loader)
            batch_size = loader.batch_size
            num_samples = len(loader.dataset)
            bs_last_batch = (
                num_samples % batch_size
                if (num_samples % self.train_batch_size != 0)
                else self.train_batch_size
            )  # this assumes drop_last=False
            running_val_losses = {}

            with torch.no_grad():
                for b_idx, batch in enumerate(tqdm(loader)):
                    if self.use_amp:
                        with autocast("cuda", dtype=torch.float16):
                            val_output = self.model.validation_step(batch)
                    else:
                        val_output = self.model.validation_step(batch)
                    losses = val_output["losses"]

                    for evaluator in evaluators:
                        evaluator.evaluate_batch(val_output, batch)
                    bs_here = batch_size if b_idx != num_batches - 1 else bs_last_batch
                    for loss_name, loss_value in losses.items():
                        if loss_name not in running_val_losses.keys():
                            running_val_losses[loss_name] = 0.0
                        running_val_losses[loss_name] += (
                            loss_value.item() * bs_here
                        ) / num_samples

            for evaluator in evaluators:
                evaluator.log_epoch(epoch)

            for loss_name, avg_loss in running_val_losses.items():
                mlflow.log_metric(f"{dataset_name}/{loss_name}", avg_loss, step=epoch)

        self.model.train()
