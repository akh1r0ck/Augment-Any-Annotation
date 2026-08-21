import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from saicinpainting.training.trainers.base import BaseInpaintingTrainingModule


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG_DIR = REPOSITORY_ROOT / "src" / "lama" / "configs" / "training"


class ManualOptimizationSmokeModel(BaseInpaintingTrainingModule):
    """Exercise the migrated BaseInpaintingTrainingModule without LaMa datasets."""

    def __init__(self):
        LightningModule.__init__(self)
        self.automatic_optimization = False
        self.generator = torch.nn.Linear(1, 1)
        self.discriminator = torch.nn.Linear(1, 1)
        self.config = SimpleNamespace(
            losses=SimpleNamespace(adversarial=SimpleNamespace(weight=1))
        )
        self.gradient_clip_val = 1.0
        self.average_generator = False

    def configure_optimizers(self):
        return [
            torch.optim.Adam(self.generator.parameters()),
            torch.optim.Adam(self.discriminator.parameters()),
        ]

    def _do_step(self, batch, batch_idx, mode="train", optimizer_idx=None, extra_val_key=None):
        layer = self.generator if optimizer_idx == 0 else self.discriminator
        loss = layer(batch[0]).pow(2).mean()
        return {
            "loss": loss,
            "log_info": {f"{mode}_loss_{optimizer_idx}": loss.detach()},
        }

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)


class Lightning2CompatibilityTest(unittest.TestCase):
    def test_manual_optimization_steps_both_optimizers(self):
        model = ManualOptimizationSmokeModel()
        trainer = Trainer(
            max_steps=2,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        trainer.fit(model)

        self.assertEqual(model.global_step, 2)

    def test_big_lama_config_builds_lightning2_trainer(self):
        os.environ.setdefault("TORCH_HOME", "/tmp/torch-home")
        with initialize_config_dir(version_base="1.1", config_dir=str(TRAINING_CONFIG_DIR)):
            config = compose(config_name="big-lama")

        trainer_kwargs = OmegaConf.to_container(config.trainer.kwargs, resolve=True)
        trainer_kwargs.pop("gradient_clip_val", None)
        trainer_kwargs.update(
            accelerator="cpu",
            devices=1,
            strategy="auto",
            limit_train_batches=1,
            limit_val_batches=0,
            num_sanity_val_steps=0,
        )

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint = ModelCheckpoint(
                dirpath=checkpoint_dir,
                **config.trainer.checkpoint_kwargs,
            )
            trainer = Trainer(
                callbacks=[checkpoint],
                logger=False,
                enable_progress_bar=False,
                **trainer_kwargs,
            )

        self.assertEqual(trainer.strategy.__class__.__name__, "SingleDeviceStrategy")


if __name__ == "__main__":
    unittest.main()
