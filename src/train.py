import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
import torch.nn as nn
import torchvision.transforms as transforms
from model import VeS
from data import VAAPairedDataset
from transformers import AutoProcessor
import numpy as np
import matplotlib.pyplot as plt
import psutil
import gc
import warnings
from pathlib import Path
import yaml
import argparse
import time
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description='Train audio-visual model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    return parser.parse_args()
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def make_vaani_collate_fn(audio_processor):
    """
    Collate function for the VAANI paired dataset.

    • Stacks (already processed) image tensors.
    • Pads & tokenises raw audio waveforms.
    • Keeps unclipped raw audio for (future) visualisation/debug.
    """

    def collate(batch):
        # --- unzip batch ---------------------------------------------------
        raw_audio_list = [item["audio"] for item in batch]          # list(np.ndarray)
        sr_list        = [item["sampling_rate"] for item in batch]
        image_tensor   = torch.stack([item["image"] for item in batch])

        processed = audio_processor(
            raw_audio_list,
            sampling_rate=sr_list[0],   # VAANI audio is 16 kHz throughout
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
    
        return {
                "input_values"  : processed.input_values,   # (B, L)
                "attention_mask": processed.attention_mask, # (B, L)
                "image"         : image_tensor,             # (B, 3, 224, 224)

                # raw components (optional, e.g. viz)
                "raw_audio"     : raw_audio_list,
                "sampling_rate" : sr_list,
            }
    return collate


class VeSTrainer:
    """Simple trainer for the VeS model with the VAANI paired dataset."""

    def __init__(self, config: dict):
        super().__init__()

        # ----------------------------- config -----------------------------
        self.cfg_train = config.get("training", {})
        self.cfg_wandb = config.get("wandb", {})

        self.device   = torch.device(
            self.cfg_train.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size                 = self.cfg_train.get("batch_size", 64)
        self.num_epochs                 = self.cfg_train.get("num_epochs", 1)
        self.steps_per_epoch            = self.cfg_train.get("steps_per_epoch", 1000)
        self.gradient_accumulation      = self.cfg_train.get("gradient_accumulation_steps", 1)
        self.checkpoint_every_steps     = self.cfg_train.get("checkpoint_every_steps", 2000)
        self.learning_rate              = self.cfg_train.get("learning_rate", 1e-4)

        self.output_dir = Path(self.cfg_train.get("output_dir", "checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ----------------------------- wandb -------------------------------
        self.use_wandb = self.cfg_wandb.get("enabled", False)
        
        
        # ----------------------------- logging ----------------------------
        logging.basicConfig(
            filename=str(self.output_dir / "training.log"),
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        if self.use_wandb:
            self._init_wandb(config)

        # ----------------------------- data --------------------------------
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        print("Initializing dataset...get ready bitches")

        dataset = VAAPairedDataset()

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=make_vaani_collate_fn(self.processor),
            num_workers=self.cfg_train.get("num_workers", 4),
            pin_memory=True,
        )
        print("pffft aight")
        
        # ----------------------------- model/optim -------------------------
        self.model = VeS(use_amp=self.cfg_train.get("use_amp", True)).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.learning_rate))

        #total_steps = self.num_epochs * self.steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20000, 
            T_mult=2,
            eta_min=1e-6
        )

        # ----------------------------- misc --------------------------------
        self.global_step = 0
        self.best_loss   = float("inf")
        
        # wandb model watching (after model and optimizer are created)
        if self.use_wandb and self.cfg_wandb.get("watch_model", False):
            wandb.watch(self.model, log="all", log_freq=self.cfg_wandb.get("log_freq", 10))

    def _init_wandb(self, config: dict):
        """Initialize Weights & Biases logging."""
        wandb_config = self.cfg_wandb
        
        # Prepare config to log to wandb
        config_to_log = {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "learning_rate": self.learning_rate,
            "gradient_accumulation_steps": self.gradient_accumulation,
            "device": str(self.device),
            "model_config": config.get("model", {}),
            "data_config": config.get("data", {}),
        }
        
        wandb.init(
            project=wandb_config.get("project", "ves-training"),
            name=wandb_config.get("name"),  # None will auto-generate
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", ""),
            config=config_to_log,
        )
        
        self.logger.info("Initialized wandb logging")

    # -------------------------------------------------------------------------
    # Check-pointing helpers
    # -------------------------------------------------------------------------

    def _ckpt_path(self, epoch: int, step: int) -> Path:
        return self.output_dir / f"checkpoint_epoch{epoch}_step{step}.pt"

    def save_checkpoint(self, epoch: int, step: int):
        ckpt = {
            "epoch": epoch,
            "step": step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        temp = self._ckpt_path(epoch, step).with_suffix(".temp.pt")
        torch.save(ckpt, temp)
        temp.rename(self._ckpt_path(epoch, step))
        self.logger.info(f"Saved checkpoint – epoch {epoch}, step {step}")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_losses = []
            steps_per_epoch = len(self.dataloader)
            pbar = tqdm(enumerate(self.dataloader), total=steps_per_epoch, desc=f"Epoch {epoch}")

            accumulation_counter = 0

            for step, batch in pbar:
                if step >= self.steps_per_epoch:
                    break  # bound the epoch length for IterableDataset

                audio  = batch["input_values"].to(self.device)
                images = batch["image"].to(self.device)

                outputs = self.model(audio, images, attention_mask=batch["attention_mask"].to(self.device))
                loss    = outputs["loss"] / self.gradient_accumulation
                if step % 100 == 0:
                    print(outputs['audio_feats'].shape)
                    print(outputs['visual_feats'].shape)
                loss.backward()
                
                accumulation_counter += 1

                if accumulation_counter % self.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                loss_val = loss.item() * self.gradient_accumulation
                epoch_losses.append(loss_val)
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

                # Log to wandb
                if self.use_wandb and self.global_step % self.cfg_wandb.get("log_freq", 10) == 0:
                    wandb.log({
                        "train/loss": loss_val,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step": self.global_step,
                    }, step=self.global_step)

                if self.global_step % self.checkpoint_every_steps == 0 and self.global_step != 0:
                    self.save_checkpoint(epoch, self.global_step)

                self.global_step += 1

            # --- epoch end --------------------------------------------------
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            self.logger.info(f"Epoch {epoch} – mean loss {mean_loss:.4f}")
            
            # Log epoch-level metrics to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch/mean_loss": mean_loss,
                    "epoch/epoch": epoch,
                }, step=self.global_step)
                
            self.save_checkpoint(epoch, self.global_step)

        print("Training completed!")
        
        # Finish wandb run
        if self.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    print("config loaded")
    trainer = VeSTrainer(config)
    print("trainer initialized")
    trainer.train()