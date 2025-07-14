# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from transformers import (
    HubertModel, 
    AutoProcessor, 
    AutoTokenizer, 
    AutoModel,
    BitsAndBytesConfig
)
import math
warnings.filterwarnings("ignore")
from torch.cuda.amp import autocast
from peft import (
    LoraConfig, 
    get_peft_model,
    TaskType,
)
import torch._dynamo
import numpy as np
from peft import prepare_model_for_kbit_training
import os 
os.environ["HIP_VISIBLE_DEVICES"] = "0"


class AudioEmbedder(nn.Module):

    def __init__(self, embedding_dim=256, hubert_name="ntu-spml/distilhubert"):
        super().__init__()
        self.hubert = AutoModel.from_pretrained(
            hubert_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        lora_cfg = LoraConfig(
            r=128,
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            bias="none",
        )

        #self.hubert = get_peft_model(self.hubert, lora_cfg)
        self.hubert.gradient_checkpointing_enable()

        self.projection1 = nn.Linear(self.hubert.config.hidden_size, 256)
        self.layer_norm = nn.LayerNorm(256)
        self.projection2 = nn.Linear(256, embedding_dim)
        self.downsample_factor = self._compute_downsample_factor()
        print(f"Downsample factor: {self.downsample_factor}")
            
        for param in self.hubert.parameters():
            param.requires_grad = True
        # Projection always trainable 

        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True
        for param in self.layer_norm.parameters():
            param.requires_grad = True

        print(f"AudioEmbedder initialized with HuBERT and LoRA.")
    
    def _compute_downsample_factor(self):
        """
        downsampling factor = product of all stride values in the convs.
        """
        downsample_factor = 1
        if hasattr(self.hubert, 'feature_extractor'):
            for layer in self.hubert.feature_extractor.conv_layers:
                if hasattr(layer, 'conv'):
                    downsample_factor *= layer.conv.stride[0]
        else:
            downsample_factor = 320
        return downsample_factor
    
    def _downsample_attention_mask(self, attention_mask: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Downsample the attention mask to match the output sequence length.
        """
        if attention_mask is None:
            return None
        
        # Always use adaptive pooling - works for all cases
        mask_float = attention_mask.float().unsqueeze(1)  # (B, 1, input_length)
        downsampled_mask = F.adaptive_avg_pool1d(mask_float, target_length)  # (B, 1, target_length)
        downsampled_mask = (downsampled_mask.squeeze(1) > 0.5).long()  # (B, target_length)
        
        return downsampled_mask
    
    @torch._dynamo.disable()
    def forward(self, audio_input: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # Convert input to bfloat16 to match the quantized model
        #audio_input = audio_input.to(dtype=torch.bfloat16)
        #if attention_mask is not None:
        #    attention_mask = attention_mask.to(dtype=torch.bfloat16)
        assert attention_mask is not None, "Attention mask is required"
        hubert_out = self.hubert(
            audio_input,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state                    # (B, Na, H)
        
   
        REDUCTION = 2
        hubert_out = hubert_out.transpose(1, 2)                # (B, H, Na)
        hubert_out = F.avg_pool1d(
            hubert_out, kernel_size=REDUCTION, stride=REDUCTION
        )                                                      # (B, H, Na//2)
        hubert_out = hubert_out.transpose(1, 2)                # (B, Na//2, H)

        if attention_mask is not None:

            mask_ds = self._downsample_attention_mask(
                attention_mask, hubert_out.size(1) * REDUCTION
            )
            output_attention_mask = mask_ds[:, ::REDUCTION]
            
        #else:
        #    B, Na_r, _ = hubert_out.shape
        #    output_attention_mask = torch.ones(
        #        B, Na_r, dtype=torch.long, device=hubert_out.device
        #    )
        else:
            raise ValueError("Attention mask is required")
        feats = self.layer_norm(self.projection1(hubert_out))
        feats = self.projection2(feats)
        feats = F.normalize(feats, dim=-1)                     # (B, Na//2, D)
        return feats, output_attention_mask
    
    def unfreeze_hubert(self):
        pass


class VisionEncoder(nn.Module):
    def __init__(self, embedding_dim=256, use_cached_features=False):
        super().__init__()
        self.use_cached_features = use_cached_features
        
        if not use_cached_features:
            self.model = AutoModel.from_pretrained('facebook/dinov2-large')
            for param in self.model.parameters():
                param.requires_grad = False
            hidden_size = self.model.config.hidden_size
            self.model.eval()
            print("Using DinoV2")
        else:
            # Skip loading DinoV2 when using cached features
            self.model = None
            print("Not using DinoV2")
            hidden_size = 1024  # DinoV2-large hidden size
        
        self.adapter = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ),
            num_layers=2
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.Linear(256, embedding_dim)
        )
    
    def forward_cached_features(self, cached_features):
        """Process cached DinoV2 features through adapter and projection.
        
        Args:
            cached_features: (B, N+1, 768) cached DinoV2 features
            
        Returns:
            patch_embeds: (B, N, embedding_dim) normalized patch embeddings
        """
        cached_features = cached_features.detach() 
        adapted = self.adapter(cached_features)  # (B, N+1, 768)
        patches = adapted[:, 1:]    # (B, N, 768) - skip CLS token
        patch_embeds = self.projection(patches)  # (B, N, embedding_dim)
        patch_embeds = F.normalize(patch_embeds, dim=-1)
        return patch_embeds
        
    def forward(self, x=None, cached_features=None):
        """Forward pass - either from images or cached features.
        
        Args:
            x: (B, 3, H, W) image tensors (required if not using cached features)
            cached_features: (B, N+1, 768) cached DinoV2 features (optional)
            
        Returns:
            patch_embeds: (B, N, embedding_dim) normalized patch embeddings
        """
        if cached_features is not None:
            return self.forward_cached_features(cached_features)
            
        if self.use_cached_features:
            raise ValueError("VisionEncoder configured for cached features but none provided")
            
        if x is None:
            raise ValueError("Either x or cached_features must be provided")
            
        with torch.no_grad():
            features = self.model(x).last_hidden_state  # [B, N+1, 768]

        adapted = self.adapter(features)  # Still [B, N+1, 768]
        patches = adapted[:, 1:]    # [B, N, 768]
        patch_embeds = self.projection(patches)  # [B, N, 256]
        patch_embeds = F.normalize(patch_embeds, dim=-1)
        
        return patch_embeds 



class VeS(nn.Module):
    def __init__(
        self, 
        loss_type="dense",
        use_cached_visual_features=False,
    ):
        super().__init__()

        self.visual_embedder = VisionEncoder(use_cached_features=use_cached_visual_features)  
        self.audio_embedder = AudioEmbedder(hubert_name="ntu-spml/distilhubert") #facebook/hubert-base-ls960
        self.use_cached_visual_features = use_cached_visual_features
        #self.logit_scale = nn.Parameter(torch.tensor(math.log(10)))
        #self.bias = nn.Parameter(torch.tensor(-10.0))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_type = loss_type
        assert self.loss_type in ["dense", "dense_global", "global"], "Invalid loss type"
        

        self.tv_weight = 0.1

    def compute_similarity_matrix(self, feats1, feats2):
        """
        token-level cosine similarity between feats1 and feats2.
        feats1: (B, N1, D)
        feats2: (B, N2, D)
        Returns sim: (B, N1, N2)
        """ 
        sim = torch.bmm(feats1, feats2.transpose(1, 2))
        #logit_scale_exp = torch.exp(self.logit_scale.clone())
        sim = sim * self.logit_scale.exp().clamp(max=100)
        return sim


    def compute_all_similarities_tv(self,
                                    audio_feats:  torch.Tensor,
                                    visual_feats: torch.Tensor,
                                    attention_mask: torch.Tensor):
        """
        Hard max over visual patches, mean over audio tokens
        and 
        Hard max over audio tokens, mean over visual patches
        Bidirectional max-mean aggregation (a → v and v → a).

        Parameters
        ----------
        audio_feats      : (B, Na, D)   audio-token embeddings
        visual_feats    : (B, Nv, D)   visual-patch embeddings
        attention_mask  : (B, Na)      1 for real tokens, 0 for padding

        Returns
        -------
        clip_sims  : (B, B)             symmetric similarity matrix
        token_sims : (B, B, Na, Nv)     raw token-level similarities
        """
        B = audio_feats.size(0)
        Na_audio = audio_feats.size(1)
        Na_mask = attention_mask.size(1)

        if Na_mask != Na_audio:
            print(f"Warning: Attention mask length ({Na_mask}) != audio feats length ({Na_audio})")
            raise ValueError(f"Attention mask and audio features have mismatched sequence lengths: {Na_mask} vs {Na_audio}")

        # token-level sim
        #token_sims = torch.matmul(af, vf.transpose(2, 3))           # (B, B, Na, Nv)

        token_sims = torch.einsum(
            'bnd, mvd -> bmnv',            # (B,Na,D) × (B,Nv,D)
            audio_feats, visual_feats
        ) 

        mask    = attention_mask[:, None, :, None]           # (B,1,Na,1)  broadcast

        masked_token_sims = torch.where(mask.bool(), token_sims, float('-inf'))


        # 1) audio → visual • max over patches, mean over tokens
        a2v_max = masked_token_sims.max(dim=3).values               # (B, B, Na)
        
        # Replace -inf with 0 before multiplication to avoid NaN
        a2v_max = torch.where(torch.isinf(a2v_max), torch.zeros_like(a2v_max), a2v_max)
        
        a_mask_2d = attention_mask.unsqueeze(1).float().expand(-1, B, -1)
        a2v_sum = (a2v_max * a_mask_2d).sum(dim=2)                  # (B, B)
        valid_a = a_mask_2d.sum(dim=2).clamp(min=1e-5)
        a2v_clip = a2v_sum / valid_a                                # (B, B)

        # 2) visual → audio • max over tokens, mean over patches  
        #v2a_max = masked_token_sims.max(dim=2).values               # (B, B, Nv)
        #v2a_max = torch.where(torch.isinf(v2a_max), torch.zeros_like(v2a_max), v2a_max)
        #v2a_clip = v2a_max.mean(dim=2)                              # (B, B)

        #clip_sims = 0.5 * (a2v_clip + v2a_clip)                     # (B, B)
        clip_sims = a2v_clip
        clip_sims = clip_sims * self.logit_scale.exp().clamp(max=100)

        return clip_sims, token_sims

    def compute_regularization_losses_tv(
        self,
        token_sims : torch.Tensor,        # (B, B, Na, Nv)
        attn_mask  : torch.Tensor | None  # (B, Na) or None
    ):
        """
        Regularisation =  l_nonneg  +  tv_weight * l_tv

        • l_nonneg  — pushes all (audio-tok, visual-patch) similarities ≥ 0  
        • l_tv      — scale-invariant total-variation smoothing on the *positive* pair
                    (diagonal b → b) along the audio-token axis
        """
        # (1) non-negativity pressure 
        neg_sims = token_sims.clamp(min=-20.0, max=0.0)
        l_nonneg = neg_sims.pow(2).mean()

        # Early-exit if temporal smoothing is disabled
        if getattr(self, "tv_weight", 0.0) == 0.0:
            return l_nonneg
        
        # (2) scale-invariant temporal-variation (TV) smoothing on diagonal
        B = token_sims.size(0)
        device = token_sims.device

        # a2v_max[b, bʹ, t] = max over visual patches
        a2v_max   = token_sims.max(dim=3).values            # (B, B, Na)
        pos_trace = a2v_max[torch.arange(B, device=device),
                            torch.arange(B, device=device)]  # (B, Na)

        if attn_mask is not None:
            m_valid   = attn_mask.float().to(device)        # (B, Na)
            neighbour = m_valid[:, 1:] * m_valid[:, :-1]    # (B, Na-1)

            # Scale-invariant TV: normalize differences by local magnitude
            eps = 1e-6
            # Use maximum of adjacent values as denominator for stability
            denominators = torch.maximum(
                pos_trace[:, :-1].abs(), 
                pos_trace[:, 1:].abs()
            ) + eps                                         # (B, Na-1)
            
            # Relative differences instead of absolute
            relative_diffs = ((pos_trace[:, 1:] - pos_trace[:, :-1]) / denominators).pow(2)
            l_tv  = (relative_diffs * neighbour).sum() / neighbour.sum().clamp_min(1.0)
        else:
            # Fallback without attention mask
            eps = 1e-6
            denominators = torch.maximum(
                pos_trace[:, :-1].abs(), 
                pos_trace[:, 1:].abs()
            ) + eps                                         # (B, Na-1)
            
            relative_diffs = ((pos_trace[:, 1:] - pos_trace[:, :-1]) / denominators).pow(2)
            l_tv = relative_diffs.mean()

        return l_nonneg + self.tv_weight * l_tv, l_nonneg, l_tv * self.tv_weight


    '''
    def compute_contrastive_loss(self, clip_sims: torch.Tensor, token_sims: torch.Tensor, attention_mask: torch.Tensor, audio_feats: torch.Tensor = None, visual_feats: torch.Tensor = None, global_weight: float = 0.0 ): #sigmoid
        """
        Pair-wise sigmoid contrastive loss for text-visual alignment.
        Algorithm 1 Sigmoid loss pseudo-implementation.
            1 # img_emb : image model embedding [n, dim]
            2 # txt_emb : text model embedding [n, dim]
            3 # t_prime, b : learnable temperature and bias
            4 # n : mini-batch size
            5
            6 t = exp(t_prime)
            7 zimg = l2_normalize(img_emb)
            8 ztxt = l2_normalize(txt_emb)
            9 logits = dot(zimg, ztxt.T) * t + b
            10 labels = 2 * eye(n) - ones(n) # -1 with diagonal 1
            11 l = -sum(log_sigmoid(labels * logits)) / n

        Parameters
        ----------
        clip_sims : (B, B)   cosine-similarity matrix between every text in the batch
                            and every image in the batch (higher = more similar)
        token_sims: (B, B, Nt, Nv) token-level similarity tensor (needed only for the
                    regularisation term carried over from the original code)

        Returns
        -------
        total_loss        : scalar torch tensor
        similarity_stats  : dict of useful monitoring statistics
        """
        B = clip_sims.size(0)
        labels = torch.eye(B, device=clip_sims.device) * 2 - 1  # +1 on diag,  elsewhere
        logits        = clip_sims + self.bias      # broadcast learnable bias b
        pairwise_loss = -F.logsigmoid(labels * logits).mean()

        #scaling loss for bf16 stability (if needed later)
        #pairwise_loss = pairwise_loss * 10

        # optional regularisation (unchanged from the original implementation)
        reg_loss, l_nonneg, l_tv = self.compute_regularization_losses_tv(token_sims, attention_mask)

        total_loss = pairwise_loss + reg_loss

        return total_loss, l_nonneg, l_tv, pairwise_loss, 0.0
    
    
    def compute_contrastive_loss(self, clip_similarities, token_sims, attention_mask):
        """ InfoNCE loss with regularization"""
        batch_size = clip_similarities.shape[0]
        
        
        labels = torch.arange(batch_size).to(clip_similarities.device)
        # Audio to Visual direction
        log_prob_a2v = F.log_softmax(clip_similarities, dim=1)
        losses_a2v = -log_prob_a2v[torch.arange(batch_size), labels]
        # Visual to Audio direction  
        log_prob_v2a = F.log_softmax(clip_similarities.t(), dim=1)
        losses_v2a = -log_prob_v2a[torch.arange(batch_size), labels]
        # Average both directions
        contrastive_loss = (losses_a2v + losses_v2a).mean() / 2
        reg_loss, l_nonneg, l_tv = self.compute_regularization_losses_tv(token_sims, attention_mask)    
        total_loss = contrastive_loss + reg_loss
        return total_loss, l_nonneg, l_tv
    '''
    
    def compute_contrastive_loss(self, clip_sims, token_sims, attention_mask, audio_feats=None, visual_feats=None, global_weight=0.5):
        """ InfoNCE loss with regularization + optional global mean-pooled loss"""
        batch_size = clip_sims.shape[0]
        labels = torch.arange(batch_size).to(clip_sims.device)
        if self.loss_type == "dense" or self.loss_type == "dense_global":
            # Existing token-level loss
            # Audio to Visual direction
            log_prob_a2v = F.log_softmax(clip_sims, dim=1)
            losses_a2v = -log_prob_a2v[torch.arange(batch_size), labels]
            # Visual to Audio direction  
            log_prob_v2a = F.log_softmax(clip_sims.t(), dim=1)
            losses_v2a = -log_prob_v2a[torch.arange(batch_size), labels]
            # Average both directions
            token_contrastive_loss = (losses_a2v + losses_v2a).mean() / 2
        else:
            token_contrastive_loss = 0.0
            
        global_contrastive_loss = 0.0
        if self.loss_type == "global" or self.loss_type == "dense_global":
            #masked mean 
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, Na, 1]
            audio_global = (audio_feats * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-6)  # [B, D]
            
            visual_global = visual_feats.mean(dim=1)  # [B, D]
            global_sims = torch.matmul(audio_global, visual_global.t()) * self.logit_scale.exp().clamp(max=100)  # [B, B]
            
            # Global InfoNCE loss
            log_prob_a2v_global = F.log_softmax(global_sims, dim=1)
            losses_a2v_global = -log_prob_a2v_global[torch.arange(batch_size), labels]
            log_prob_v2a_global = F.log_softmax(global_sims.t(), dim=1)
            losses_v2a_global = -log_prob_v2a_global[torch.arange(batch_size), labels]
            
            global_contrastive_loss = (losses_a2v_global + losses_v2a_global).mean() / 2
        
        contrastive_loss = (1 - global_weight) * token_contrastive_loss + global_weight * global_contrastive_loss
        
        reg_loss, l_nonneg, l_tv = self.compute_regularization_losses_tv(token_sims, attention_mask)    
        total_loss = contrastive_loss + reg_loss
        
        return total_loss, l_nonneg, l_tv, token_contrastive_loss, global_contrastive_loss
    
        
    
    def forward(self, audio_input, images=None, attention_mask=None, cached_visual_features=None):
        """
        Forward pass of VeS model.
        
        Args:
            audio_input: (B, T) raw audio waveform at 16kHz
            images: (B, 3, H, W) preprocessed image tensors (optional if using cached features)
            attention_mask: (B, T) audio attention mask
            cached_visual_features: (B, N+1, 768) cached DinoV2 features (optional)
            
        Returns:
            dict containing:
                - loss: total loss (contrastive + regularization)
                - clip_sims: (B, B) clip-level similarity matrix
                - audio_feats: (B, Na, D) audio features
                - visual_feats: (B, Nv, D) visual features
        """
        audio_feats, audio_attention_mask = self.audio_embedder(audio_input, attention_mask)
        
        if cached_visual_features is not None:
            visual_feats = self.visual_embedder(cached_features=cached_visual_features)
        elif images is not None:
            print("Cached features not provided, using images")
            visual_feats = self.visual_embedder(images)
        else:
            raise ValueError("Either images or cached_visual_features must be provided")
        clip_sims, token_sims = self.compute_all_similarities_tv(
            audio_feats, 
            visual_feats, 
            audio_attention_mask
        )

        #loss, l_nonneg, l_tv = self.compute_contrastive_loss(clip_sims, token_sims, audio_attention_mask)
        if self.loss_type == "dense":
            loss, l_nonneg, l_tv, dense_loss, global_loss = self.compute_contrastive_loss(
                clip_sims, 
                token_sims, 
                audio_attention_mask,
                audio_feats=None,     
                visual_feats=None,    
                global_weight=0.0             
            )
        elif self.loss_type == "dense_global":
            loss, l_nonneg, l_tv, dense_loss, global_loss = self.compute_contrastive_loss(
                clip_sims, 
                token_sims, 
                audio_attention_mask,
                audio_feats=audio_feats,     
                visual_feats=visual_feats,    
                global_weight=0.30             
            )
        elif self.loss_type == "global":
            loss, l_nonneg, l_tv, dense_loss, global_loss = self.compute_contrastive_loss(
                clip_sims, 
                token_sims, 
                audio_attention_mask,
                audio_feats=audio_feats,     
                visual_feats=visual_feats,    
                global_weight=1.0             
            )
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        
        return {
            'loss': loss,
            'clip_sims': clip_sims.detach(),
            'audio_feats': audio_feats.detach(),
            'visual_feats': visual_feats.detach(),
            'audio_attention_mask': audio_attention_mask.detach(),
            'token_sims': token_sims.detach(),
            'l_nonneg': l_nonneg.detach(),
            'l_tv': l_tv.detach(),
            'dense_loss': dense_loss.detach() if isinstance(dense_loss, torch.Tensor) else dense_loss,  
            'global_loss': global_loss.detach() if isinstance(global_loss, torch.Tensor) else global_loss
        }

    def unfreeze_hubert(self):
        """Unfreeze HuBERT encoder for second stage of training."""
        self.audio_embedder.unfreeze_hubert()




def dummy_inputs():
    """Fixture to create dummy audio and image inputs."""
    batch_size = 2
    audio_seq_len = 16000 * 5  # 5 seconds of audio at 16kHz
    audio_input = torch.randn(batch_size, audio_seq_len)
    images = torch.randn(batch_size, 3, 224, 224)
    
    return audio_input, images




if __name__ == "__main__":
    print("Testing VeS with random inputs...")

    print("\n=== Testing with HuBERT frozen (Stage 1) ===")
    model = VeS(freeze_hubert_initially=True).to("cuda")

    audio_input, images = dummy_inputs()
    audio_input = audio_input.to("cuda")
    images = images.to("cuda")

    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (Stage 1): {trainable_before:,}")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(audio_input, images)
    print(f"Visual features shape: {outputs['visual_feats'].shape}")
    print(f"Audio features shape: {outputs['audio_feats'].shape}")
    print(f"Clip similarities shape: {outputs['clip_sims'].shape}")

    print("\n=== Testing HuBERT unfreezing (Stage 2) ===")
    model.unfreeze_hubert()
    
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (Stage 2): {trainable_after:,}")
    print(f"Newly unfrozen params: {trainable_after - trainable_before:,}")
    
    model.train()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(audio_input, images)
        print(f"Visual features shape: {outputs['visual_feats'].shape}")
        print(f"Audio features shape: {outputs['audio_feats'].shape}")
        print(f"Clip similarities shape: {outputs['clip_sims'].shape}")
