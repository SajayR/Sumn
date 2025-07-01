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
import bitsandbytes as bnb
#from splus import SPlus
from viz import VeSVisualizer
import torch.multiprocessing as mp
mp.set_start_method('fork', force=True)  # Explicit is better than implicit


warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.98) 
torch.compiler.reset()  
#torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import os 
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_COMPILE_DEBUG"] = "1"

class VeSTrainer:
    """Simple trainer for the VeS model with the VAANI paired dataset."""

    def __init__(self, config: dict, use_cached_visual_features=False, cached_features_base_path=None):
        super().__init__()

        # ----------------------------- config -----------------------------
        self.cfg_train = config.get("training", {})
        self.cfg_wandb = config.get("wandb", {})
        self.use_cached_visual_features = use_cached_visual_features
        self.cached_features_base_path = cached_features_base_path

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
        dataset = VAAPairedDataset(cached_features_base_path=self.cached_features_base_path)

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
            num_workers=10,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            #prefetch_factor=6,
            shuffle=True,
            generator=self.data_generator, 
            worker_init_fn=worker_init_fn, 
            #pin_memory_device="cuda" 
        )
        self.steps_per_epoch = len(self.dataloader)
        
        self.model = VeS(loss_type="dense", use_cached_visual_features=self.use_cached_visual_features).to(self.device)
        #self.model.visual_embedder.fuse_lora()
        #torch._dynamo.reset()
        #self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=False)#, fullgraph=True)#, dynamic=False)
        self.model.train()

        
        #self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.learning_rate), fused=True)
        self.optimizer = bnb.optim.Adam8bit(self.model.parameters(), lr=float(self.learning_rate)) #gives like 600 MBs in savings
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
        self.epoch_step = 0  # Track step within current epoch
        self.best_loss   = float("inf")
        if self.use_wandb:
            self._init_wandb(config)
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

    def save_checkpoint(self, epoch: int, step: int, current_epoch_step: int = None):
        # Use the provided current_epoch_step or fall back to tracked epoch_step
        epoch_step = current_epoch_step if current_epoch_step is not None else self.epoch_step
        print(f"DEBUG: Saving checkpoint with epoch_step={epoch_step} (provided={current_epoch_step}, tracked={self.epoch_step})")
        
        ckpt = {
            "epoch": epoch,
            "step": step,
            "global_step": self.global_step,
            "epoch_step": epoch_step,  # Step within current epoch for mid-epoch resume
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
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
        self.logger.info(f"Saved checkpoint – epoch {epoch}, step {step}, epoch_step {epoch_step}")

    def load_checkpoint(self, checkpoint_path: str | Path):
        """Load checkpoint and restore exact training state."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Restore model and training states
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        
        # Restore training progress
        self.global_step = ckpt.get("global_step", ckpt["step"])
        self.current_epoch = ckpt["epoch"]
        self.epoch_step = ckpt.get("epoch_step", 0)  # Restore step within epoch
        self.best_loss = ckpt.get("best_loss", float("inf"))
        print(f"DEBUG: Loaded checkpoint with epoch_step={self.epoch_step} from saved checkpoint")
        
        # Restore random states for deterministic continuation
        # Handle RNG states carefully to avoid device/type issues
        try:
            if "torch_rng_state" in ckpt:
                rng_state = ckpt["torch_rng_state"]
                # Ensure the RNG state is on CPU and is a ByteTensor
                if hasattr(rng_state, 'cpu'):
                    rng_state = rng_state.cpu()
                if not isinstance(rng_state, torch.ByteTensor):
                    if hasattr(rng_state, 'byte'):
                        rng_state = rng_state.byte()
                torch.set_rng_state(rng_state)
        except Exception as e:
            print(f"Warning: Could not restore torch RNG state: {e}")
            
            
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])
        
        try:
            if "data_generator_state" in ckpt:
                generator_state = ckpt["data_generator_state"]
                # Ensure the generator state is on CPU and is a ByteTensor
                if hasattr(generator_state, 'cpu'):
                    generator_state = generator_state.cpu()
                if not isinstance(generator_state, torch.ByteTensor):
                    if hasattr(generator_state, 'byte'):
                        generator_state = generator_state.byte()
                self.data_generator.set_state(generator_state)
        except Exception as e:
            print(f"Warning: Could not restore data generator RNG state: {e}")
            # Fallback: Reset data generator with the original seed
            self.data_generator.manual_seed(self.data_seed)
        
        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}, epoch_step {self.epoch_step}")
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def print_trainable_params(self, stage_name=""):
        """Print current trainable parameter status for debugging staged training."""
        print(f"\n=== Trainable Parameter Status {stage_name} ===")
        
        # Count parameters by component
        audio_hubert_base_params = sum(p.numel() for n, p in self.model.audio_embedder.hubert.named_parameters() if p.requires_grad and 'lora_' not in n)
        audio_hubert_base_params_frozen = sum(p.numel() for n, p in self.model.audio_embedder.hubert.named_parameters() if not p.requires_grad and 'lora_' not in n)
        audio_hubert_lora_params = sum(p.numel() for n, p in self.model.audio_embedder.hubert.named_parameters() if p.requires_grad and 'lora_' in n)
        audio_proj_params = sum(p.numel() for n, p in self.model.audio_embedder.named_parameters() 
                               if p.requires_grad and 'hubert' not in n)
        
        vision_base_params = sum(p.numel() for n, p in self.model.visual_embedder.model.named_parameters() 
                                if p.requires_grad and 'lora_' not in n ) if self.model.visual_embedder.model is not None else 0
        vision_base_params_frozen = sum(p.numel() for n, p in self.model.visual_embedder.model.named_parameters() if not p.requires_grad and 'lora_' not in n) if self.model.visual_embedder.model is not None else 0
        #vision_lora_params = sum(p.numel() for n, p in self.model.visual_embedder.model.named_parameters() 
        #                        if p.requires_grad and 'lora_' in n)
        vision_proj_params = sum(p.numel() for n, p in self.model.visual_embedder.named_parameters() 
                                if p.requires_grad and 'model' not in n)
        
        other_params = sum(p.numel() for n, p in self.model.named_parameters() 
                          if p.requires_grad and 'audio_embedder' not in n and 'visual_embedder' not in n)
        
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"Audio HuBERT Base: {audio_hubert_base_params:,} params {'(WARNING: SHOULD BE FROZEN!)' if audio_hubert_base_params > 0 else '(FROZEN)'}")
        print(f"Audio HuBERT Base Frozen: {audio_hubert_base_params_frozen:,} params (FROZEN)")
        print(f"Audio HuBERT LoRA: {audio_hubert_lora_params:,} params (TRAINABLE)")
        print(f"Audio Projections: {audio_proj_params:,} params (TRAINABLE)")
        print(f"Vision Base Model: {vision_base_params:,} params {'(WARNING: SHOULD BE FROZEN!)' if vision_base_params > 0 else '(FROZEN)'}")
        print(f"Vision Base Model Frozen: {vision_base_params_frozen:,} params (FROZEN)")
        #print(f"Vision LoRA: {vision_lora_params:,} params (TRAINABLE)")
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
            
            # Deterministic reshuffling: Set epoch-specific seed
            # This ensures different shuffling per epoch but reproducible across runs
            epoch_seed = self.data_seed + epoch * 1000  # Multiply by 1000 to ensure different seeds
            self.data_generator.manual_seed(epoch_seed)
            print(f"Epoch {epoch}: Using deterministic seed {epoch_seed} for data shuffling")
            
            # Also update worker seeds for this epoch
            def epoch_worker_init_fn(worker_id):
                worker_seed = epoch_seed + worker_id
                np.random.seed(worker_seed)
                random.seed(worker_seed)
                torch.manual_seed(worker_seed)
            
            # Create a new DataLoader with the epoch-specific seed
            # This is necessary because we can't change the generator state of an existing DataLoader
            epoch_dataloader = DataLoader(
                self.dataloader.dataset,
                batch_size=self.batch_size,
                num_workers=10,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True,
                shuffle=True,
                generator=self.data_generator,
                worker_init_fn=epoch_worker_init_fn,
                #pin_memory_device="cuda"
            )
            
            # Determine if we need to skip steps (for mid-epoch resume)
            steps_to_skip = 0
            if epoch == self.current_epoch and self.epoch_step > 0:
                steps_to_skip = self.epoch_step
                print(f"Resuming from epoch {epoch}, skipping first {steps_to_skip} steps")
            
            pbar = tqdm(enumerate(epoch_dataloader), total=self.steps_per_epoch, desc=f"Epoch {epoch}")
            
            # Reset epoch_step at the beginning of a new epoch
            if epoch != self.current_epoch:
                self.epoch_step = 0

            accumulation_counter = 0
            '''print("\n=== LoRA Parameter Verification ===")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"TRAINABLE: {name} - {param.numel():,} params")
                elif 'lora_' in name:
                    print(f"WARNING: LoRA param not trainable: {name}")
            print("=== End Verification ===\n")'''
            for step, batch in pbar:
                # Skip already processed steps when resuming mid-epoch
                if step < steps_to_skip:
                    pbar.set_postfix({"status": f"Skipping processed step {step}/{steps_to_skip-1}"})
                    continue
                    
                #if step<200:
                #    continue
                if step >= self.steps_per_epoch:
                    break  # bound the epoch length for IterableDataset

                audio = batch["audio"].to(self.device, non_blocking=True)
                attention_mask = batch["audio_attention_mask"].to(self.device, non_blocking=True)
                
                # Use cached visual features if available, otherwise use images
                if "cached_visual_features" in batch:
                    cached_features = batch["cached_visual_features"].to(self.device, non_blocking=False)
                    with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        outputs = self.model(audio, attention_mask=attention_mask, cached_visual_features=cached_features)
                else:
                    images = batch["image"].to(self.device, non_blocking=True)
                    with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        outputs = self.model(audio, images=images, attention_mask=attention_mask)
                
                loss = outputs["loss"] / self.gradient_accumulation

                loss.backward()
                #loss.detach()
                
                accumulation_counter += 1

                if accumulation_counter % self.gradient_accumulation == 0:
                    # Compute gradient norm before clipping for monitoring
                    grad_norm, param_count = self.compute_gradient_norm()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    #torch.nn.utils.clip_grad_norm_(self.model.audio_embedder.hubert.parameters(), 0.5)
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
                if self.global_step % self.viz_every_steps == 0:# and self.global_step != 0:
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
                    to_log={
                        "train/loss": loss_val,
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch,
                        "train/step": self.global_step,
                        "train/l_nonneg": outputs["l_nonneg"].item(),
                        "train/l_tv": outputs["l_tv"].item(),
                        "train/logit_scale": self.model.logit_scale.exp().item(),
                        "train/clip_sims": outputs["clip_sims"].mean().item(),
                        "train/clip_diagonal_sims": outputs["clip_sims"].diagonal().mean().item(),
                    }
                    if "dense_loss" in outputs:
                        to_log["train/dense_loss"] = outputs["dense_loss"].item()
                    if "global_loss" in outputs:
                        to_log["train/global_loss"] = outputs["global_loss"].item() if isinstance(outputs["global_loss"], torch.Tensor) else outputs["global_loss"]
                    wandb.log(to_log, step=self.global_step)


                if self.global_step % self.checkpoint_every_steps == 0 and self.global_step != 0:
                    self.save_checkpoint(epoch, self.global_step, step + 1)

                self.global_step += 1
                self.epoch_step = step + 1  # Track current step within epoch

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
                
            self.save_checkpoint(epoch, self.global_step, self.epoch_step)

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
            "num_workers": 6, #12,
            "data_seed": 42,  # Fixed seed for deterministic data ordering
            
            # Training schedule
            "num_epochs": 3,
            
            # Optimization
            "learning_rate": 3e-4,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.1,  
            
            # Checkpointing
            "output_dir": "checkpoints",
            "checkpoint_every_steps": 20000,
            
            "viz_every_steps": 2500,
            "viz_batch_limit": 32,
        },
        "logging": {
            "level": "INFO",
            "log_file": "training.log",
        },
        "wandb": {
            "enabled": True,
            "project": "fuckaroundlol",
            "name": "large-dino",
            "log_freq": 1, 
            "watch_model": False,  
        },
    }

    # Example usage with cached features:
    # trainer = VeSTrainer(config, use_cached_visual_features=True, cached_features_base_path="/speedy/Vaani")
    
    # Or without cached features (will compute from images):
    trainer = VeSTrainer(config, use_cached_visual_features=True, cached_features_base_path="/speedy/CisStuff/cached_features/dinov2_large")
    #trainer = VeSTrainer(config, use_cached_visual_features=False)
    print("trainer initialized")
    
    # Automatically resume from the latest checkpoint if available
    resumed = trainer.auto_resume_if_available()
    if resumed:
        print("Resumed training from checkpoint")
    else:
        print("Starting fresh training")
    
    trainer.train()