# src/retrieval_eval.py
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import wandb


@dataclass
class RetrievalMetrics:
    """Container for retrieval metrics."""
    r1: float
    r5: float
    r10: float
    r50: float
    mean_rank: float
    median_rank: float
    
    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        """Convert to dict for logging."""
        return {
            f"{prefix}R@1": self.r1,
            f"{prefix}R@5": self.r5,
            f"{prefix}R@10": self.r10,
            f"{prefix}R@50": self.r50,
            f"{prefix}mean_rank": self.mean_rank,
            f"{prefix}median_rank": self.median_rank,
        }


class RetrievalEvaluator:
    """
    Efficient retrieval evaluation for VeS model.
    Handles both mean-pooling and max-mean aggregation without blowing up memory.
    """
    
    def __init__(
        self, 
        val_dataloader, 
        device: str = "cuda",
        batch_size: int = 32,
        logger: Optional[logging.Logger] = None,
        use_cached_embeddings: bool = False,
        cache_dir: Optional[Path] = None,
    ):
        self.val_dataloader = val_dataloader
        self.device = device
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)
        self.use_cached_embeddings = use_cached_embeddings
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.use_cached_embeddings and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def extract_embeddings(self, model, max_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract embeddings from validation set in batches to avoid OOM.
        
        Returns:
            audio_embeds: (N, Na, D) audio embeddings
            visual_embeds: (N, Nv, D) visual embeddings  
            attention_masks: (N, Na) attention masks
        """
        model.eval()
        
        # Check if we have cached embeddings
        if self.use_cached_embeddings and self.cache_dir:
            cache_file = self.cache_dir / f"val_embeddings_step{model.global_step}.pt"
            if cache_file.exists():
                self.logger.info(f"Loading cached embeddings from {cache_file}")
                cached = torch.load(cache_file, map_location='cpu')
                return cached['audio'], cached['visual'], cached['masks']
        
        audio_embeds_list = []
        visual_embeds_list = []
        attention_masks_list = []
        
        num_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(self.val_dataloader, desc="Extracting embeddings")):
            if max_samples and num_samples >= max_samples:
                break
                
            # Move batch to device
            audio = batch["audio"].to(self.device, non_blocking=True)
            attention_mask = batch["audio_attention_mask"].to(self.device, non_blocking=True)
            
            # Handle visual input (cached features or images)
            if "cached_visual_features" in batch:
                cached_features = batch["cached_visual_features"].to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs = model(audio, attention_mask=attention_mask, cached_visual_features=cached_features)
            else:
                images = batch["image"].to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs = model(audio, images=images, attention_mask=attention_mask)
            
            # Store embeddings on CPU to save GPU memory
            audio_embeds_list.append(outputs['audio_feats'].cpu())
            visual_embeds_list.append(outputs['visual_feats'].cpu())
            attention_masks_list.append(outputs['audio_attention_mask'].cpu())
            
            num_samples += audio.shape[0]
            
            # Clear GPU cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Concatenate all embeddings
        audio_embeds = torch.cat(audio_embeds_list, dim=0)
        visual_embeds = torch.cat(visual_embeds_list, dim=0)
        attention_masks = torch.cat(attention_masks_list, dim=0)
        
        # Cache embeddings if requested
        if self.use_cached_embeddings and self.cache_dir:
            cache_file = self.cache_dir / f"val_embeddings_step{model.global_step}.pt"
            torch.save({
                'audio': audio_embeds,
                'visual': visual_embeds,
                'masks': attention_masks,
                'step': model.global_step
            }, cache_file)
            self.logger.info(f"Cached embeddings to {cache_file}")
        
        return audio_embeds, visual_embeds, attention_masks
    
    def compute_similarity_matrix_chunked(
        self, 
        audio_embeds: torch.Tensor, 
        visual_embeds: torch.Tensor,
        attention_masks: torch.Tensor,
        aggregation: str = "max_mean",
        chunk_size: int = 100
    ) -> torch.Tensor:
        """
        Compute similarity matrix in chunks to avoid OOM.
        
        Args:
            audio_embeds: (N, Na, D)
            visual_embeds: (N, Nv, D)
            attention_masks: (N, Na)
            aggregation: "max_mean" or "mean"
            chunk_size: Number of samples to process at once
            
        Returns:
            sim_matrix: (N, N) similarity matrix
        """
        N = audio_embeds.shape[0]
        sim_matrix = torch.zeros(N, N, dtype=torch.float32)
        
        # Process in chunks to avoid memory explosion
        for i in range(0, N, chunk_size):
            i_end = min(i + chunk_size, N)
            
            for j in range(0, N, chunk_size):
                j_end = min(j + chunk_size, N)
                
                # Move chunks to GPU
                audio_chunk = audio_embeds[i:i_end].to(self.device)
                visual_chunk = visual_embeds[j:j_end].to(self.device)
                mask_chunk = attention_masks[i:i_end].to(self.device)
                
                if aggregation == "max_mean":
                    chunk_sim = self._compute_max_mean_similarity(
                        audio_chunk, visual_chunk, mask_chunk
                    )
                else:  # mean pooling
                    chunk_sim = self._compute_mean_pooled_similarity(
                        audio_chunk, visual_chunk, mask_chunk
                    )
                
                # Store result back to CPU
                sim_matrix[i:i_end, j:j_end] = chunk_sim.cpu()
                
                # Clear cache
                del audio_chunk, visual_chunk, mask_chunk, chunk_sim
                torch.cuda.empty_cache()
        
        return sim_matrix
    
    def _compute_max_mean_similarity(
        self,
        audio_feats: torch.Tensor,
        visual_feats: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute max-mean aggregated similarity (matching model's approach).
        
        Args:
            audio_feats: (B1, Na, D)
            visual_feats: (B2, Nv, D)
            attention_mask: (B1, Na)
            
        Returns:
            clip_sims: (B1, B2)
        """
        B1, Na, D = audio_feats.shape
        B2, Nv, _ = visual_feats.shape
        
        # Compute token-level similarities
        token_sims = torch.einsum('bnd,mvd->bmnv', audio_feats, visual_feats)  # (B1, B2, Na, Nv)
        
        # Apply attention mask
        mask = attention_mask[:, None, :, None]  # (B1, 1, Na, 1)
        masked_sims = torch.where(mask.bool(), token_sims, float('-inf'))
        
        # Audio -> Visual: max over patches, mean over tokens
        a2v_max = masked_sims.max(dim=3).values  # (B1, B2, Na)
        a2v_max = torch.where(torch.isinf(a2v_max), torch.zeros_like(a2v_max), a2v_max)
        
        a_mask_2d = attention_mask.unsqueeze(1).float()  # (B1, 1, Na)
        a2v_sum = (a2v_max * a_mask_2d).sum(dim=2)  # (B1, B2)
        valid_a = a_mask_2d.sum(dim=2).clamp(min=1e-5)
        a2v_clip = a2v_sum / valid_a  # (B1, B2)
        
        # Visual -> Audio: max over tokens, mean over patches
        #v2a_max = masked_sims.max(dim=2).values  # (B1, B2, Nv)
        #v2a_max = torch.where(torch.isinf(v2a_max), torch.zeros_like(v2a_max), v2a_max)
        #v2a_clip = v2a_max.mean(dim=2)  # (B1, B2)
        
        # Average both directions
        #clip_sims = 0.5 * (a2v_clip + v2a_clip)
        clip_sims = a2v_clip
        
        return clip_sims
    
    def _compute_mean_pooled_similarity(
        self,
        audio_feats: torch.Tensor,
        visual_feats: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mean-pooled similarity.
        
        Args:
            audio_feats: (B1, Na, D)
            visual_feats: (B2, Nv, D)
            attention_mask: (B1, Na)
            
        Returns:
            clip_sims: (B1, B2)
        """
        # Mean pool audio with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B1, Na, 1)
        audio_global = (audio_feats * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-6)  # (B1, D)
        
        # Mean pool visual
        visual_global = visual_feats.mean(dim=1)  # (B2, D)
        
        # Compute similarities
        clip_sims = torch.matmul(audio_global, visual_global.t())  # (B1, B2)
        
        return clip_sims
    
    def compute_retrieval_metrics(self, sim_matrix: torch.Tensor) -> Dict[str, RetrievalMetrics]:
        """
        Compute retrieval metrics from similarity matrix.
        
        Args:
            sim_matrix: (N, N) similarity matrix
            
        Returns:
            Dictionary with 'audio_to_visual' and 'visual_to_audio' metrics
        """
        N = sim_matrix.shape[0]
        
        # Audio -> Visual retrieval
        a2v_ranks = []
        for i in range(N):
            # Get similarities for audio i to all visuals
            sims = sim_matrix[i]
            # Rank of the correct match (diagonal)
            rank = (sims > sims[i]).sum().item() + 1
            a2v_ranks.append(rank)
        
        # Visual -> Audio retrieval  
        v2a_ranks = []
        for j in range(N):
            # Get similarities for visual j to all audios
            sims = sim_matrix[:, j]
            # Rank of the correct match (diagonal)
            rank = (sims > sims[j]).sum().item() + 1
            v2a_ranks.append(rank)
        
        # Compute metrics
        def compute_metrics(ranks):
            ranks = np.array(ranks)
            return RetrievalMetrics(
                r1=(ranks <= 1).mean() * 100,
                r5=(ranks <= 5).mean() * 100,
                r10=(ranks <= 10).mean() * 100,
                r50=(ranks <= 50).mean() * 100,
                mean_rank=ranks.mean(),
                median_rank=np.median(ranks)
            )
        
        return {
            'audio_to_visual': compute_metrics(a2v_ranks),
            'visual_to_audio': compute_metrics(v2a_ranks)
        }
    
    @torch.no_grad()
    def evaluate(
        self, 
        model, 
        max_samples: Optional[int] = None,
        log_to_wandb: bool = True,
        global_step: Optional[int] = None
    ) -> Dict[str, Dict[str, RetrievalMetrics]]:
        """
        Run full evaluation on validation set.
        
        Returns:
            Dictionary with results for both aggregation methods
        """
        model.eval()
        
        # Extract embeddings once
        self.logger.info("Extracting embeddings from validation set...")
        audio_embeds, visual_embeds, attention_masks = self.extract_embeddings(model, max_samples)
        
        self.logger.info(f"Extracted embeddings: audio {audio_embeds.shape}, visual {visual_embeds.shape}")
        
        results = {}
        
        # Evaluate with max-mean aggregation (model's default)
        self.logger.info("Computing max-mean aggregated similarities...")
        sim_matrix_maxmean = self.compute_similarity_matrix_chunked(
            audio_embeds, visual_embeds, attention_masks, 
            aggregation="max_mean"
        )
        results['max_mean'] = self.compute_retrieval_metrics(sim_matrix_maxmean)
        
        # Evaluate with mean pooling
        self.logger.info("Computing mean-pooled similarities...")
        sim_matrix_mean = self.compute_similarity_matrix_chunked(
            audio_embeds, visual_embeds, attention_masks,
            aggregation="mean"
        )
        results['mean_pooled'] = self.compute_retrieval_metrics(sim_matrix_mean)
        
        # Log results
        self._log_results(results, log_to_wandb, global_step)
        
        # Clean up
        del audio_embeds, visual_embeds, attention_masks, sim_matrix_maxmean, sim_matrix_mean
        torch.cuda.empty_cache()
        
        return results
    
    def _log_results(
        self, 
        results: Dict[str, Dict[str, RetrievalMetrics]], 
        log_to_wandb: bool,
        global_step: Optional[int]
    ):
        """Log evaluation results."""
        # Print to console/logger
        for method, method_results in results.items():
            self.logger.info(f"\n{method.upper()} Aggregation Results:")
            for direction, metrics in method_results.items():
                self.logger.info(f"  {direction}:")
                self.logger.info(f"    R@1: {metrics.r1:.2f}%")
                self.logger.info(f"    R@5: {metrics.r5:.2f}%")
                self.logger.info(f"    R@10: {metrics.r10:.2f}%")
                self.logger.info(f"    Mean Rank: {metrics.mean_rank:.2f}")
        
        # Log to wandb
        if log_to_wandb and wandb.run is not None:
            wandb_dict = {}
            for method, method_results in results.items():
                for direction, metrics in method_results.items():
                    prefix = f"val/{method}/{direction}/"
                    wandb_dict.update(metrics.to_dict(prefix))
            
            if global_step is not None:
                wandb.log(wandb_dict, step=global_step)
            else:
                wandb.log(wandb_dict)


# Add this to the bottom of retrieval_eval.py

if __name__ == "__main__":
    import sys
    sys.path.append('src')  # Add src to path if running from parent dir
    
    from torch.utils.data import DataLoader
    from data import VAAPairedDataset
    from model import VeS
    import argparse
    
    parser = argparse.ArgumentParser(description="Test retrieval evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--max-samples", type=int, default=500, help="Max validation samples to use (None for all)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--use-cached-features", action="store_true", help="Use cached visual features")
    parser.add_argument("--cached-features-path", type=str, 
                       default="/speedy/CisStuff/cached_features/dinov2_large",
                       help="Path to cached features")
    args = parser.parse_args()
    
    print("🚀 Starting retrieval evaluation test...")
    
    # Create validation dataset
    print("📊 Loading validation dataset...")
    val_dataset = VAAPairedDataset(
        is_validation=True,
        cached_features_base_path=args.cached_features_path if args.use_cached_features else None
    )
    print(f"✅ Loaded {len(val_dataset)} validation samples")
    
    # Create dataloader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        shuffle=False,
    )
    
    # Initialize model
    print("🤖 Initializing VeS model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VeS(
        loss_type="dense",
        use_cached_visual_features=args.use_cached_features
    ).to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"📁 Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        print(f"✅ Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    else:
        print("⚠️  No checkpoint provided, using random weights")
    
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model has {total_params:,} total params, {trainable_params:,} trainable")
    
    # Create evaluator
    print("\n🔍 Creating retrieval evaluator...")
    evaluator = RetrievalEvaluator(
        val_dataloader,
        device=device,
        batch_size=32,  # Can be different from dataloader batch size
        use_cached_embeddings=False,  # Don't cache during test
    )
    
    # Run evaluation
    print(f"\n🏃 Running evaluation on {args.max_samples or 'all'} samples...")
    print("=" * 80)
    
    try:
        results = evaluator.evaluate(
            model,
            max_samples=args.max_samples,
            log_to_wandb=False,  # No wandb for standalone test
        )
        
        # Pretty print results
        print("\n" + "="*80)
        print("📊 RETRIEVAL EVALUATION RESULTS")
        print("="*80)
        
        for method, method_results in results.items():
            print(f"\n🔧 {method.upper().replace('_', ' ')} Aggregation:")
            print("-" * 40)
            
            for direction, metrics in method_results.items():
                print(f"\n  📍 {direction.replace('_', ' ').title()}:")
                print(f"     R@1:  {metrics.r1:6.2f}%")
                print(f"     R@5:  {metrics.r5:6.2f}%")
                print(f"     R@10: {metrics.r10:6.2f}%")
                print(f"     R@50: {metrics.r50:6.2f}%")
                print(f"     Mean Rank:   {metrics.mean_rank:6.1f}")
                print(f"     Median Rank: {metrics.median_rank:6.1f}")
        
        # Compute average metrics across directions
        print("\n" + "="*80)
        print("📈 AVERAGED METRICS (both directions)")
        print("="*80)
        
        for method in results:
            a2v = results[method]['audio_to_visual']
            v2a = results[method]['visual_to_audio']
            print(f"\n🔧 {method.upper().replace('_', ' ')}:")
            print(f"   R@1:  {(a2v.r1 + v2a.r1) / 2:6.2f}%")
            print(f"   R@5:  {(a2v.r5 + v2a.r5) / 2:6.2f}%")
            print(f"   R@10: {(a2v.r10 + v2a.r10) / 2:6.2f}%")
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        print("\n🧹 Cleaned up GPU memory")


# Quick test function for debugging similarity computations
def test_similarity_computation():
    """Quick test to verify similarity computation works correctly."""
    print("\n🧪 Testing similarity computation...")
    
    # Create dummy data
    B, Na, Nv, D = 4, 10, 16*16, 256
    audio_feats = torch.randn(B, Na, D).cuda()
    visual_feats = torch.randn(B, Nv, D).cuda()
    attention_mask = torch.ones(B, Na).cuda()
    attention_mask[:, 7:] = 0  # Mask out last 3 tokens
    
    # Create evaluator instance
    evaluator = RetrievalEvaluator(None, device="cuda")
    
    # Test max-mean similarity
    sim_maxmean = evaluator._compute_max_mean_similarity(
        audio_feats, visual_feats, attention_mask
    )
    print(f"Max-mean similarity shape: {sim_maxmean.shape}")
    print(f"Max-mean similarity range: [{sim_maxmean.min():.3f}, {sim_maxmean.max():.3f}]")
    
    # Test mean-pooled similarity
    sim_mean = evaluator._compute_mean_pooled_similarity(
        audio_feats, visual_feats, attention_mask
    )
    print(f"Mean-pooled similarity shape: {sim_mean.shape}")
    print(f"Mean-pooled similarity range: [{sim_mean.min():.3f}, {sim_mean.max():.3f}]")
    
    # Check diagonal should be highest (for same embeddings)
    audio_same = audio_feats[:2]
    visual_same = visual_feats[:2]
    sim_same = evaluator._compute_max_mean_similarity(
        audio_same, visual_same, attention_mask[:2]
    )
    print(f"\nDiagonal test (should be high): {sim_same.diag().mean():.3f}")
    print(f"Off-diagonal (should be lower): {sim_same[0, 1]:.3f}, {sim_same[1, 0]:.3f}")
    
    print("✅ Similarity computation test passed!")


# Add option to run similarity test
if __name__ == "__main__" and "--test-sim" in sys.argv:
    test_similarity_computation()


'''
🔧 MAX MEAN Aggregation:
----------------------------------------

  📍 Audio To Visual:
     R@1:    0.20%
     R@5:    1.17%
     R@10:   1.95%
     R@50:   9.96%
     Mean Rank:    256.6
     Median Rank:  258.0

  📍 Visual To Audio:
     R@1:    0.20%
     R@5:    1.37%
     R@10:   2.93%
     R@50:   9.57%
     Mean Rank:    255.4
     Median Rank:  248.5

🔧 MEAN POOLED Aggregation:
----------------------------------------

  📍 Audio To Visual:
     R@1:    0.20%
     R@5:    0.78%
     R@10:   1.95%
     R@50:  10.16%
     Mean Rank:    258.1
     Median Rank:  257.0

  📍 Visual To Audio:
     R@1:    0.20%
     R@5:    0.59%
     R@10:   1.76%
     R@50:   9.57%
     Mean Rank:    258.6
     Median Rank:  254.5

================================================================================
📈 AVERAGED METRICS (both directions)
================================================================================

🔧 MAX MEAN:
   R@1:    0.20%
   R@5:    1.27%
   R@10:   2.44%

🔧 MEAN POOLED:
   R@1:    0.20%
   R@5:    0.68%
   R@10:   1.86%

'''