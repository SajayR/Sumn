#### train.py ####

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
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
import numpy as np
import matplotlib.pyplot as plt
import psutil
import gc
import warnings
from pathlib import Path
import time
import random
import pickle
#import bitsandbytes as bnb
#from splus import SPlus
from viz import VeSVisualizer
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.95) 
#torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


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
        self.gradient_accumulation      = self.cfg_train.get("gradient_accumulation_steps", 1)
        self.checkpoint_every_steps     = self.cfg_train.get("checkpoint_every_steps", 2000)
        self.learning_rate              = self.cfg_train.get("learning_rate", 1e-4)
        self.visualize_every_steps      = self.cfg_train.get("visualize_every_steps", 10)
        self.viz_every_steps            = self.cfg_train.get("viz_every_steps", 10)
        self.hubert_unfreeze_steps      = self.cfg_train.get("hubert_unfreeze_steps", None)
        
        # Set deterministic seed for reproducible data ordering
        self.data_seed = self.cfg_train.get("data_seed", 42)
        self.set_seeds(self.data_seed)

        self.output_dir = Path(self.cfg_train.get("output_dir", "checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize visualizer
        REDUCTION = 2
        self.visualizer = VeSVisualizer(
            out_dir=self.output_dir / "viz",
            token_hz=25,            # 20 ms per token
            max_samples_per_call=20,
            reduction=REDUCTION,
        )

        self.use_wandb = self.cfg_wandb.get("enabled", False)

        logging.basicConfig(
            filename=str(self.output_dir / "training.log"),
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        self.cfg_data = config.get("data", {})
        dataset = VAAPairedDataset()

        # Create DataLoader with deterministic worker seeding
        def worker_init_fn(worker_id):
            # Each worker gets a unique but deterministic seed
            worker_seed = self.data_seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        # Create generator for deterministic shuffling
        self.data_generator = torch.Generator()
        self.data_generator.manual_seed(self.data_seed)

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            prefetch_factor=6,
            shuffle=True,
            generator=self.data_generator, 
            worker_init_fn=worker_init_fn, 
            pin_memory_device="cuda" 
        )
        self.steps_per_epoch = len(self.dataloader)
        
        freeze_hubert = self.hubert_unfreeze_steps is not None and self.hubert_unfreeze_steps > 0
        self.model = VeS(freeze_hubert_initially=freeze_hubert).to(self.device)
        #self.model.visual_embedder.fuse_lora()
        #torch._dynamo.reset()
        #self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=False)#, fullgraph=True)#, dynamic=False)
        self.model.train()
        self.hubert_unfrozen = not freeze_hubert

        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.learning_rate), fused=True)
        #self.optimizer = bnb.optim.Adam8bit(self.model.parameters(), lr=float(self.learning_rate)) #gives like 600 MBs in savings
        #self.optimizer = SPlus(self.model.parameters(), lr=float(self.learning_rate)) #adds like 600MBs in whatever the opposite of savings is
        
        total_steps = len(self.dataloader)*self.num_epochs // self.gradient_accumulation
        warmup_steps = int(self.cfg_train.get("warmup_ratio", 0.1) * total_steps)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,  # Half cosine cycle
            last_epoch=-1
        )
        
        print(f"Scheduler setup: {total_steps} total steps, {warmup_steps} warmup steps")

        self.global_step = 0
        self.current_epoch = 0
        self.best_loss   = float("inf")
        if self.use_wandb:
            self._init_wandb(config)
            wandb.log({
                "train/stage": 1 if freeze_hubert else 0,  # Stage 1: LoRA + projections only, Stage 0: all trainable
                "train/hubert_unfrozen": 0 if freeze_hubert else 1,
            }, step=0)
        if self.use_wandb and self.cfg_wandb.get("watch_model", False):
            wandb.watch(self.model, log="all", log_freq=self.cfg_wandb.get("log_freq", 10))

    def set_seeds(self, seed: int):
        """Set all random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # For deterministic CUDA operations (may impact performance)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        print(f"Set all random seeds to {seed} for deterministic training")

    def _init_wandb(self, config: dict):
        """Initialize Weights & Biases logging."""
        wandb_config = self.cfg_wandb
        
        # Prepare config to log to wandb
        config_to_log = {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            #"steps_per_epoch": self.steps_per_epoch,
            "learning_rate": self.learning_rate,
            "gradient_accumulation_steps": self.gradient_accumulation,
            "device": str(self.device),
            "model_config": config.get("model", {}),
            "data_config": config.get("data", {}),
            "data_seed": self.data_seed,
            #"num_workers": self.cfg_train.get("num_workers", 4),
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

    def find_latest_checkpoint(self) -> Path | None:
        """Find the checkpoint with the highest step number in the output directory."""
        if not self.output_dir.exists():
            return None
        
        checkpoint_files = list(self.output_dir.glob("checkpoint_epoch*_step*.pt"))
        if not checkpoint_files:
            return None
        
        # Extract step numbers and find the maximum
        latest_step = -1
        latest_checkpoint = None
        
        for ckpt_file in checkpoint_files:
            # Parse filename: checkpoint_epoch{epoch}_step{step}.pt
            try:
                filename = ckpt_file.stem  # Remove .pt extension
                parts = filename.split('_')
                step_part = next(part for part in parts if part.startswith('step'))
                step_num = int(step_part[4:])  # Remove 'step' prefix
                
                if step_num > latest_step:
                    latest_step = step_num
                    latest_checkpoint = ckpt_file
            except (ValueError, StopIteration, IndexError):
                # Skip malformed filenames
                continue
        
        return latest_checkpoint

    def auto_resume_if_available(self) -> bool:
        """Automatically find and load the latest checkpoint if available.
        
        Returns:
            bool: True if a checkpoint was loaded, False otherwise
        """
        latest_checkpoint = self.find_latest_checkpoint()
        if latest_checkpoint:
            print(f"Found latest checkpoint: {latest_checkpoint}")
            self.load_checkpoint(latest_checkpoint)
            return True
        else:
            print("No existing checkpoints found, starting training from scratch")
            return False

    def save_checkpoint(self, epoch: int, step: int):
        ckpt = {
            "epoch": epoch,
            "step": step,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "hubert_unfrozen": self.hubert_unfrozen,
            "best_loss": self.best_loss,
            # Save random states for exact resumption
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "data_generator_state": self.data_generator.get_state(),
            "data_seed": self.data_seed,
        }
        temp = self._ckpt_path(epoch, step).with_suffix(".temp.pt")
        torch.save(ckpt, temp)
        temp.rename(self._ckpt_path(epoch, step))
        self.logger.info(f"Saved checkpoint – epoch {epoch}, step {step}")

    def load_checkpoint(self, checkpoint_path: str | Path):
        """Load checkpoint and restore exact training state."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model and training states
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        
        # Restore training progress
        self.global_step = ckpt.get("global_step", ckpt["step"])
        self.current_epoch = ckpt["epoch"]
        self.hubert_unfrozen = ckpt.get("hubert_unfrozen", False)
        self.best_loss = ckpt.get("best_loss", float("inf"))
        
        # Restore random states for deterministic continuation
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if "cuda_rng_state" in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])
        if "data_generator_state" in ckpt:
            self.data_generator.set_state(ckpt["data_generator_state"])
        
        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        print(f"HuBERT unfrozen: {self.hubert_unfrozen}")
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

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
                                if p.requires_grad and 'lora_' in n)
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


    def train(self):
        # Print initial parameter status
        self.print_trainable_params("(Initial State)")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            #self.optimizer.train()

            epoch_losses = []
            
            pbar = tqdm(enumerate(self.dataloader), total=self.steps_per_epoch, desc=f"Epoch {epoch}")

            accumulation_counter = 0
            '''print("\n=== LoRA Parameter Verification ===")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"TRAINABLE: {name} - {param.numel():,} params")
                elif 'lora_' in name:
                    print(f"WARNING: LoRA param not trainable: {name}")
            print("=== End Verification ===\n")'''
            for step, batch in pbar:
                #if step<200:
                #    continue
                if step >= self.steps_per_epoch:
                    break  # bound the epoch length for IterableDataset

                # Check if we need to unfreeze HuBERT
                if (not self.hubert_unfrozen and 
                    self.hubert_unfreeze_steps is not None and 
                    self.global_step >= self.hubert_unfreeze_steps):
                    
                    print(f"\n Reached step {self.global_step} - Unfreezing HuBERT encoder!")
                    self.model.unfreeze_hubert()
                    
                    # Recreate optimizer with newly unfrozen parameters
                    old_lr = self.scheduler.get_last_lr()[0] if hasattr(self, 'scheduler') else self.learning_rate
                    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(old_lr))
                    
                    # Set initial_lr for all param groups (required for scheduler)
                    for group in self.optimizer.param_groups:
                        group['initial_lr'] = self.learning_rate
                    
                    # Recreate scheduler with the same state
                    total_steps = self.num_epochs * self.steps_per_epoch // self.gradient_accumulation
                    warmup_steps = int(self.cfg_train.get("warmup_ratio", 0.1) * total_steps)
                    self.scheduler = get_cosine_schedule_with_warmup(
                        self.optimizer,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=total_steps,
                        num_cycles=0.5,
                        last_epoch=self.global_step - 1  # Continue from current step
                    )
                    
                    self.hubert_unfrozen = True
                    self.print_trainable_params("(After HuBERT Unfreezing)")
                    self.logger.info(f"Unfroze HuBERT at step {self.global_step}")
                    
                    # Log to wandb
                    if self.use_wandb:
                        wandb.log({
                            "train/hubert_unfrozen": 1,
                            "train/stage": 2,  # Stage 2: Full model training
                        }, step=self.global_step)

                audio  = batch["audio"].to(self.device, non_blocking=True)
                images = batch["image"].to(self.device, non_blocking=True)
                
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(audio, images, attention_mask=batch["audio_attention_mask"].to(self.device))
                    loss    = outputs["loss"] / self.gradient_accumulation

                loss.backward()
                
                accumulation_counter += 1

                if accumulation_counter % self.gradient_accumulation == 0:
                    # Compute gradient norm before clipping for monitoring
                    grad_norm, param_count = self.compute_gradient_norm()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.audio_embedder.hubert.parameters(), 0.5)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Log gradient norm to wandb
                    if self.use_wandb and self.global_step % self.cfg_wandb.get("log_freq", 10) == 0:
                        wandb.log({
                            "gradients/grad_norm": grad_norm,
                            "gradients/param_count": param_count,
                        }, step=self.global_step)

                # ---------------------------------------------------------
                #   periodic visualisation
                # ---------------------------------------------------------
                if self.global_step % self.viz_every_steps == 0 and self.global_step != 0:
                    matplotlib_figures = self.visualizer.visualize_batch(
                        batch,                       
                        outputs,
                        step=self.global_step,
                    )
                    
                    # Log matplotlib figures to wandb
                    if self.use_wandb and matplotlib_figures:
                        wandb_images = {}
                        for basename, fig in matplotlib_figures:
                            wandb_images[f"heatmaps/{basename}"] = wandb.Image(fig)
                            plt.close(fig)  # Free memory
                        
                        wandb.log(wandb_images, step=self.global_step)

                loss_val = loss.item() * self.gradient_accumulation
                epoch_losses.append(loss_val)
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

                # Log to wandb
                if self.use_wandb and self.global_step % self.cfg_wandb.get("log_freq", 10) == 0:
                    current_lr = self.scheduler.get_last_lr()[0] if self.scheduler.get_last_lr() else self.learning_rate
                    wandb.log({
                        "train/loss": loss_val,
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch,
                        "train/step": self.global_step,
                        "bias": self.model.bias.item(),
                        "logit_scale": self.model.logit_scale.item(),
                    }, step=self.global_step)

                if self.global_step % self.checkpoint_every_steps == 0 and self.global_step != 0:
                    self.save_checkpoint(epoch, self.global_step)

                self.global_step += 1

            # --- epoch end --------------------------------------------------
            self.current_epoch = epoch + 1
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
    # Hardcoded configuration (previously from config.yaml)
    config = {
        "training": {
            # Device and basic settings
            "device": "cuda",
            "use_amp": True,
            
            # Data settings
            "batch_size": 54,
            "num_workers": 12,
            "data_seed": 42,  # Fixed seed for deterministic data ordering
            
            # Training schedule
            "num_epochs": 10,
          
            
            # Optimization
            "learning_rate": 3e-4,
            "gradient_accumulation_steps": 5,
            "warmup_ratio": 0.1,  
            "hubert_unfreeze_steps": 2000,  
            
            # Checkpointing
            "output_dir": "checkpoints",
            "checkpoint_every_steps": 2000,
            
            "viz_every_steps": 2500,
            "viz_batch_limit": 32,
        },
        "logging": {
            "level": "INFO",
            "log_file": "training.log",
        },
        "wandb": {
            "enabled": True,
            "project": "VeS-Wtf",
            "name": "WtfMode",
            "log_freq": 1, 
            "watch_model": False,  
        },
    }

    trainer = VeSTrainer(config)
    print("trainer initialized")
    
    # Automatically resume from the latest checkpoint if available
    resumed = trainer.auto_resume_if_available()
    if resumed:
        print("Resumed training from checkpoint")
    else:
        print("Starting fresh training")
    
    trainer.train()