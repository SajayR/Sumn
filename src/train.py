#### train.py ####

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
import torch.nn as nn
import torchvision.transforms as transforms
from model import VeS, _encode_representations, _loss_on_reprs, _run_audio_backward, _run_vision_backward, _clip_sims_blockwise
from data import VAAPairedDataset
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
import numpy as np
import matplotlib.pyplot as plt
import psutil
import gc
import warnings
from pathlib import Path
import time
#import bitsandbytes as bnb
#from splus import SPlus
from viz import VeSVisualizer
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()


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
        #self.steps_per_epoch            = self.cfg_train.get("steps_per_epoch", 1000)
        self.checkpoint_every_steps     = self.cfg_train.get("checkpoint_every_steps", 2000)
        self.learning_rate              = self.cfg_train.get("learning_rate", 1e-4)
        self.visualize_every_steps      = self.cfg_train.get("visualize_every_steps", 10)
        self.viz_every_steps            = self.cfg_train.get("viz_every_steps", 10)
        self.hubert_unfreeze_steps      = self.cfg_train.get("hubert_unfreeze_steps", None)

        self.output_dir = Path(self.cfg_train.get("output_dir", "checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        #self.vis_out_dir = self.output_dir / "visualizations"
        #self.vis_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = VeSVisualizer(
            out_dir=self.output_dir / "viz",
            token_hz=50,            # 20 ms per token
            max_samples_per_call=20,
        )

        
        # ----------------------------- wandb -------------------------------
        self.use_wandb = self.cfg_wandb.get("enabled", False)
        
        
        # ----------------------------- logging ----------------------------
        logging.basicConfig(
            filename=str(self.output_dir / "training.log"),
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)


        # ----------------------------- data --------------------------------
        #self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        #print("Initializing dataset...get ready bitches")

        # Get data configuration
        self.cfg_data = config.get("data", {})
        dataset = VAAPairedDataset()

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            #collate_fn=make_vaani_collate_fn(self.processor),
            num_workers=self.cfg_train.get("num_workers", 4),
            pin_memory=True,
            persistent_workers=False,
            drop_last=True,
            prefetch_factor=4,
        )
        self.steps_per_epoch = len(self.dataloader)
        
        # ----------------------------- model/optim -------------------------
        # Initialize model with staged training if configured
        freeze_hubert = self.hubert_unfreeze_steps is not None and self.hubert_unfreeze_steps > 0
        self.model = VeS(freeze_hubert_initially=freeze_hubert).to(self.device)
        #self.model.visual_embedder.fuse_lora()
        #self.model = torch.compile(self.model, mode="max-autotune")#, fullgraph=True, dynamic=False)
        self.model.train()
        
        # Track if we've unfrozen HuBERT
        self.hubert_unfrozen = not freeze_hubert
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.learning_rate))
        #self.optimizer = bnb.optim.Adam8bit(self.model.parameters(), lr=float(self.learning_rate)) #gives like 600 MBs in savings
        #self.optimizer = SPlus(self.model.parameters(), lr=float(self.learning_rate)) #adds like 600MBs in whatever the opposite of savings is
        
        # Calculate total steps for cosine schedule with warmup
        total_steps = len(self.dataloader)
        warmup_steps = int(self.cfg_train.get("warmup_ratio", 0.1) * total_steps)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,  # Half cosine cycle
            last_epoch=-1
        )
        
        print(f"Scheduler setup: {total_steps} total steps, {warmup_steps} warmup steps")

        # ----------------------------- misc --------------------------------
        # -------------------------------------------------------------
        # Auto-resume: look for an existing checkpoint to resume from
        # -------------------------------------------------------------

        self.start_epoch = 0          # epoch to start / resume from
        self.start_step_in_epoch = 0  # "step" offset within current epoch

        self.global_step = 0

        if self.cfg_train.get("auto_resume", True):
            latest_ckpt = self._find_latest_checkpoint()
            if latest_ckpt is not None:
                self._load_checkpoint(latest_ckpt)

        # keep track of best loss seen so far (not currently used)
        self.best_loss   = float("inf")
        if self.use_wandb:
            self._init_wandb(config)
            # Log initial training stage
            wandb.log({
                "train/stage": 1 if freeze_hubert else 0,  # Stage 1: LoRA + projections only, Stage 0: all trainable
                "train/hubert_unfrozen": 0 if freeze_hubert else 1,
            }, step=0)
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
            #"steps_per_epoch": self.steps_per_epoch,
            "learning_rate": self.learning_rate,
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

    def compute_gradient_norm(self):
        """Compute the gradient norm for monitoring training stability."""
        total_norm = 0.0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2) if param_count > 0 else 0.0
        return total_norm, param_count

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

    def print_trainable_params(self, stage_name=""):
        """Print current trainable parameter status for debugging staged training."""
        print(f"\n=== Trainable Parameter Status {stage_name} ===")
        
        # Count parameters by component
        audio_hubert_params = sum(p.numel() for n, p in self.model.audio_embedder.hubert.named_parameters() if p.requires_grad)
        audio_proj_params = sum(p.numel() for n, p in self.model.audio_embedder.named_parameters() 
                               if p.requires_grad and 'hubert' not in n)
        
        vision_base_params = sum(p.numel() for n, p in self.model.visual_embedder.model.named_parameters() 
                                if p.requires_grad and 'lora_' not in n)
        vision_lora_params = sum(p.numel() for n, p in self.model.visual_embedder.model.named_parameters() 
                                if p.requires_grad and 'lora' in n)
        vision_proj_params = sum(p.numel() for n, p in self.model.visual_embedder.named_parameters() 
                                if p.requires_grad and 'model' not in n)
        
        other_params = sum(p.numel() for n, p in self.model.named_parameters() 
                          if p.requires_grad and 'audio_embedder' not in n and 'visual_embedder' not in n)
        
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"Audio HuBERT: {audio_hubert_params:,} params {'(TRAINABLE)' if audio_hubert_params > 0 else '(FROZEN)'}")
        print(f"Audio Projections: {audio_proj_params:,} params (TRAINABLE)")
        print(f"Vision Base Model: {vision_base_params:,} params {'(WARNING: SHOULD BE FROZEN!)' if vision_base_params > 0 else '(FROZEN)'}")
        print(f"Vision LoRA: {vision_lora_params:,} params (TRAINABLE)")
        print(f"Vision Projections: {vision_proj_params:,} params (TRAINABLE)")
        print(f"Other (logit_scale, bias): {other_params:,} params (TRAINABLE)")
        print(f"Total Trainable: {total_trainable:,} / {total_params:,} ({100 * total_trainable / total_params:.2f}%)")
        print("=== End Status ===\n")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
# VeSTrainer.train  —  Grad-Cache edition
# -------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
#                         Grad-Cache TRAIN LOOP
# ---------------------------------------------------------------------------
    def train(self):
        self.print_trainable_params("(Initial State)")

        micro_bs  = self.cfg_train.get("micro_batch_size", 32)
        encode_bs = self.cfg_train.get("encode_micro_bs", micro_bs)
        blk_pairs = self.cfg_train.get("pair_block", 32)      # for _clip_sims_blockwise

        for epoch in range(self.num_epochs):
            epoch_losses = []
            pbar = tqdm(
                enumerate(self.dataloader), total=self.steps_per_epoch,
                desc=f"Epoch {epoch}"
            )

            for step, batch in pbar:
                if step >= self.steps_per_epoch:
                    break

                # --------------------- HuBERT staged training ----------------
                if (not self.hubert_unfrozen and
                    self.hubert_unfreeze_steps is not None and
                    self.global_step >= self.hubert_unfreeze_steps):
                    print(f"\n>>> Step {self.global_step}: unfreezing HuBERT")
                    self.model.unfreeze_hubert()
                    lr_now = self.scheduler.get_last_lr()[0]
                    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr_now)
                    tot_steps  = self.num_epochs * self.steps_per_epoch
                    warm      = int(self.cfg_train.get("warmup_ratio", 0.1)*tot_steps)
                    self.scheduler = get_cosine_schedule_with_warmup(
                        self.optimizer, warm, tot_steps, num_cycles=0.5,
                        last_epoch=self.global_step-1
                    )
                    self.hubert_unfrozen = True
                    self.print_trainable_params("(After HuBERT Unfreezing)")
                    if self.use_wandb:
                        wandb.log({"train/hubert_unfrozen":1,"train/stage":2},
                                step=self.global_step)

                # --------------------- Grad-Cache STEP-1  (enc fwd) ----------
                F, G, extra = _encode_representations(
                    self.model, batch, self.device,
                    encode_micro_bs=encode_bs, store_on_cpu=False
                )

                # --------------------- similarity & loss (no huge tensor) ----
                scale = torch.exp(self.model.logit_scale)
                clip_sims, token_diag = _clip_sims_blockwise(
                    F, G, extra["a_mask"], scale, blk=blk_pairs
                )

                B = clip_sims.size(0)
                labels = torch.eye(B, device=clip_sims.device)*2 - 1
                logits = clip_sims + self.model.bias
                loss   = -torch.nn.functional.logsigmoid(labels*logits).mean()

                loss.backward()                       # fills F.grad / G.grad
                grad_F, grad_G = F.grad.detach(), G.grad.detach()

                # --------------------- Grad-Cache STEP-2  (enc back) ---------
                self.optimizer.zero_grad(set_to_none=True)
                _run_audio_backward(self.model, batch, grad_F, micro_bs, self.device)
                _run_vision_backward(self.model, batch, grad_G, micro_bs, self.device)

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                # --------------------- bookkeeping ---------------------------
                loss_val = float(loss.detach())
                epoch_losses.append(loss_val)
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

                if self.use_wandb and self.global_step % self.cfg_wandb.get("log_freq",10)==0:
                    wandb.log({
                        "train/loss": loss_val,
                        "train/lr":   self.scheduler.get_last_lr()[0],
                        "grad_norm":  float(grad_norm),
                        "train/step": self.global_step,
                        "train/epoch":epoch,
                    }, step=self.global_step)

                # ---- visualiser needs only diagonal token sims --------------
                if self.global_step % self.viz_every_steps == 0:
                    viz_out = {
                        "loss": loss.detach(),
                        "clip_sims": clip_sims.detach(),
                        "token_sims": token_diag.detach(),   # (B,Na,Nv)
                        "audio_feats": F.detach(),
                        "visual_feats": G.detach(),
                        "audio_attention_mask": extra["a_mask"].detach(),
                    }
                    figs = self.visualizer.visualize_batch(batch, viz_out,
                                                        step=self.global_step)
                    if self.use_wandb and figs:
                        wandb.log({f"heat/{n}": wandb.Image(f) for n,f in figs},
                                step=self.global_step)

                if (self.global_step % self.checkpoint_every_steps == 0
                        and self.global_step):
                    self.save_checkpoint(epoch, self.global_step)

                self.global_step += 1

            # --------------------- end-of-epoch ------------------------------
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            self.logger.info(f"Epoch {epoch} – mean loss {mean_loss:.4f}")
            if self.use_wandb:
                wandb.log({"epoch/mean_loss": mean_loss, "epoch/epoch": epoch},
                        step=self.global_step)
            self.save_checkpoint(epoch, self.global_step)

        print("Training completed!")
        if self.use_wandb:
            wandb.finish()


    # -----------------------------------------------------------------
    # Checkpoint RESUME helpers
    # -----------------------------------------------------------------

    def _find_latest_checkpoint(self):
        """Return Path to most recent checkpoint or None if none exist."""
        ckpt_files = list(self.output_dir.glob("checkpoint_epoch*_step*.pt"))
        ckpt_files = [p for p in ckpt_files if not p.name.endswith(".temp.pt")]
        if not ckpt_files:
            return None

        # Pick checkpoint with the largest step index (parse after '_step')
        def _step_num(p: Path):
            try:
                return int(p.stem.split("_step")[1])
            except (IndexError, ValueError):
                return -1

        latest = max(ckpt_files, key=_step_num)
        return latest

    def _load_checkpoint(self, ckpt_path: Path):
        """Load model/optim/scheduler state and update bookkeeping vars."""
        self.logger.info(f"Resuming from checkpoint {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        # Restore state
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])

        # Book-keeping
        self.global_step = int(ckpt.get("step", 0))
        self.start_epoch = int(ckpt.get("epoch", 0))
        self.start_step_in_epoch = self.global_step % self.steps_per_epoch

        # If we are exactly at an epoch boundary (i.e., all batches of the epoch
        # are complete), move to the next epoch to avoid repeating work.
        if self.start_step_in_epoch == 0 and self.global_step != 0:
            self.start_epoch += 1

        # Work out HuBERT unfrozen status after loading
        if self.hubert_unfreeze_steps is not None and self.global_step < self.hubert_unfreeze_steps:
            self.hubert_unfrozen = False
        else:
            self.hubert_unfrozen = True

        self.logger.info(
            f"Checkpoint loaded – epoch {self.start_epoch}, global step {self.global_step}, "
            f"start_step_in_epoch {self.start_step_in_epoch}"
        )

        print(f"[Auto-resume] Loaded checkpoint {ckpt_path} | epoch {self.start_epoch} | "
              f"global step {self.global_step}")


if __name__ == "__main__":
    # Hardcoded configuration (previously from config.yaml)
    config = {
        "training": {
            # Device and basic settings
            "device": "cuda",
            "use_amp": True,
            
            # Data settings
            "micro_batch_size": 32,        # fits comfortably in GPU RAM
            "batch_size": 256,
            "num_workers": 4,
            
            # Training schedule
            "num_epochs": 3,
          
            
            # Optimization
            "learning_rate": 3e-4,
            "warmup_ratio": 0.1,  
            "hubert_unfreeze_steps": 0,  
            
            # Checkpointing
            "output_dir": "checkpoints",
            "checkpoint_every_steps": 2000,
            "auto_resume": False,
            
            "viz_every_steps": 5000,
            "viz_batch_limit": 32,
        },
        "logging": {
            "level": "INFO",
            "log_file": "training.log",
        },
        "wandb": {
            "enabled": False,
            "project": "VeS",
            "name": "Staged_Training",
            "log_freq": 1, 
            "watch_model": False,  
        },
    }
    
    print("Using hardcoded configuration")
    trainer = VeSTrainer(config)
    print("trainer initialized")
    trainer.train()