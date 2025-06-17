# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from transformers import (
    HubertModel, 
    AutoProcessor, 
    AutoTokenizer, 
    AutoModel
)
import math
warnings.filterwarnings("ignore")
from torch.cuda.amp import autocast
from peft import (
    LoraConfig, 
    get_peft_model,
    TaskType,
)

class AudioEmbedder(nn.Module):

    def __init__(self, embedding_dim=256, hubert_name="ntu-spml/distilhubert", freeze_hubert_initially=True):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(hubert_name)
        self.hubert.gradient_checkpointing_enable()

        self.projection1 = nn.Linear(self.hubert.config.hidden_size, 256)
        self.layer_norm = nn.LayerNorm(256)
        self.projection2 = nn.Linear(256, embedding_dim)
        self.downsample_factor = self._compute_downsample_factor()
        #print(f"Downsample factor: {self.downsample_factor}")
        
        # Initially freeze HuBERT parameters if requested (for staged training)
        self.hubert_frozen = freeze_hubert_initially
        for param in self.hubert.parameters():
            param.requires_grad = not freeze_hubert_initially
            
        # Projection layers are always trainable from the start
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True
        for param in self.layer_norm.parameters():
            param.requires_grad = True
            
        if freeze_hubert_initially:
            print(f"AudioEmbedder initialized with HuBERT FROZEN")
        else:
            print(f"AudioEmbedder initialized with HuBERT UNFROZEN")
    
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
        
        Args:
            attention_mask: (B, L) input attention mask
            target_length: desired output length
            
        Returns:
            downsampled_mask: (B, target_length) output attention mask
        """
        if attention_mask is None:
            return None
            
        batch_size = attention_mask.size(0)
        input_length = attention_mask.size(1)
        
        # Method 1: Simple downsampling by taking every nth element
        # This works well when the downsampling is uniform
        if input_length // self.downsample_factor == target_length:
            # Perfect match - use stride-based downsampling
            downsampled_mask = attention_mask[:, ::self.downsample_factor][:, :target_length]
        else:
            # Method 2: Adaptive pooling to handle any length mismatch
            # Reshape to (B, 1, L) for adaptive pooling
            mask_float = attention_mask.float().unsqueeze(1)
            
            # Use adaptive average pooling to downsample
            downsampled_mask = F.adaptive_avg_pool1d(mask_float, target_length)
            
            # Convert back to binary mask (threshold at 0.5)
            downsampled_mask = (downsampled_mask.squeeze(1) > 0.5).long()
        
        return downsampled_mask
        
    def forward(self, audio_input: torch.Tensor, attention_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_input: (B, L) processed audio features
            attention_mask: (B, L) attention mask for audio tokens (input length)
            
        Returns:
            audio_feats: (B, Na, D) where Na is the OUTPUT sequence length
            output_attention_mask: (B, Na) attention mask for output tokens
        """
        hubert_outputs = self.hubert(
            audio_input, 
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hubert_output = hubert_outputs.last_hidden_state  # (B, Na, H)
        output_length = hubert_output.size(1)
        
        # Downsample the attention mask to match output length
        if attention_mask is not None:
            output_attention_mask = self._downsample_attention_mask(attention_mask, output_length)
        else:
            
            batch_size = hubert_output.size(0)
            output_attention_mask = torch.ones(
                batch_size, output_length,
                dtype=torch.long,
                device=hubert_output.device
            )
        audio_feats = self.projection2(self.layer_norm(self.projection1(hubert_output)))
        audio_feats = F.normalize(audio_feats, dim=-1)
        return audio_feats, output_attention_mask
    
    def unfreeze_hubert(self):
        """Unfreeze HuBERT parameters for fine-tuning after warmup period."""
        if self.hubert_frozen:
            print("Unfreezing HuBERT parameters...")
            for param in self.hubert.parameters():
                param.requires_grad = True
            self.hubert_frozen = False
            print(f"HuBERT parameters unfrozen. Total trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        else:
            print("HuBERT parameters are already unfrozen")

class VisionEncoder(nn.Module):

    def __init__(self, embedding_dim=256, patch_dropout_prob=0.1, lora_rank=16, lora_alpha=32):
        super().__init__()

        self.model = AutoModel.from_pretrained('facebook/dinov2-with-registers-base')
        self.model.gradient_checkpointing_enable()
        for param in self.model.parameters():
            param.requires_grad = False

        lora_target_modules = [
            "attention.attention.query",
            "attention.attention.key",
            "attention.attention.value",
            "attention.output.dense",
            # ff
            # "mlp.fc1",
            # "mlp.fc2",
        ]

        lora_config = LoraConfig(
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules, 
            #bias="none",          
            #modules_to_save=None,  
        )

        self.model = get_peft_model(self.model, lora_config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"VisionEncoder - Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")
        self.projection1 = nn.Linear(self.model.config.hidden_size, 256)
        self.layer_norm = nn.LayerNorm(256)
        self.projection2 = nn.Linear(256, embedding_dim)
        #self.patch_dropout_rate = patch_dropout_prob
        #self.patch_dropout = self.patch_dropout_layer

        for name, param in self.model.named_parameters():
            if 'lora_' not in name: 
                param.requires_grad = False
            else:
                param.requires_grad = True
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True

    def fuse_lora(self):
        self.model = self.model.merge_and_unload() 

    def patch_dropout_layer(self, x: torch.Tensor, drop_p: float):
        """
        x : (B, N, D) 
        """
        if not self.training or drop_p == 0:
            return x

        B, N, D = x.shape
        keep_mask = (torch.rand(B, N, 1, device=x.device) > drop_p)
        x = x * keep_mask 
        keep_counts = keep_mask.sum(dim=1, keepdim=True).clamp_min(1)
        x = x * N / keep_counts
        return x

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W), e.g. (B,3,224,224) image batch
        Returns:
            visual_feats: (B, Nv, D)
                Nv = number of visual tokens
                D  = embedding_dim
        """
        #print("Dino input shape: ", x.shape)
        outputs = self.model(pixel_values=x,return_dict=True,output_attentions=False, output_hidden_states=False)
        patches = outputs.last_hidden_state[:,5:, :] #5 cause 1 is the cls token, 4 are registers
        feats = self.projection2(self.layer_norm(self.projection1(patches)))
        #if self.training:
          #  feats = self.patch_dropout(feats, self.patch_dropout_rate)
        feats = F.normalize(feats, dim=-1)
        #print("Dino output shape: ", feats.shape)
        return feats


# ------------------------------------------------------------
#  helpers (put near the top of train.py)
# ------------------------------------------------------------
def _detach_repr(t: torch.Tensor) -> torch.Tensor:
    """
    Returns a view of `t` that **requires_grad**, but has no history.
    """
    return t.detach().requires_grad_(True)

'''
@torch.no_grad()
def _encode_representations(model, batch, device):
    """
    Graph-less forward.  Returns (F, G, extra) where

        F  – audio token reprs  (B, N_a, D)
        G  – visual patch reprs (B, N_v, D)
        extra – dict with 'mask' etc. for later reuse
    """
    audio  = batch["audio"].to(device)
    images = batch["image"].to(device)
    mask   = batch["audio_attention_mask"].to(device)
    print(f"Audio shape: {audio.shape}")
    model.eval()                               # ❶  no dropout
    with torch.no_grad():
        a_repr, a_mask = model.audio_embedder(audio, mask)
        v_repr         = model.visual_embedder(images)

    model.train()                              # ❷  back to training mode
    return (
        _detach_repr(a_repr),                  # F
        _detach_repr(v_repr),                  # G
        {"a_mask": a_mask},
    )'''

# ------------------------------------------------------------------
#  encode the *virtual* batch in micro-batches  (Step-1)
# ------------------------------------------------------------------
@torch.no_grad()
def _encode_representations(
    model,
    batch,
    device,
    encode_micro_bs: int | None = None,
    store_on_cpu: bool = False,          # set True if still tight on VRAM
):
    """
    Returns
    -------
      F  : (B, N_a, D)  audio-token reps (requires_grad = True)
      G  : (B, N_v, D)  visual-patch reps (requires_grad = True)
      extra : {"a_mask": (B, N_a)}
    """
    audio  = batch["audio"]
    images = batch["image"]
    mask   = batch["audio_attention_mask"]

    B = audio.size(0)
    encode_micro_bs = encode_micro_bs or B         # full batch if not given

    a_chunks, m_chunks, v_chunks = [], [], []

    model.eval()                                   # disable dropout
    for s in range(0, B, encode_micro_bs):
        e = min(s + encode_micro_bs, B)

        a_rep, a_mask = model.audio_embedder(
            audio[s:e].to(device), mask[s:e].to(device)
        )
        v_rep = model.visual_embedder(images[s:e].to(device))

        if store_on_cpu:                           # optional CPU offload
            a_rep  = a_rep.cpu()
            v_rep  = v_rep.cpu()
            a_mask = a_mask.cpu()

        a_chunks.append(a_rep)
        m_chunks.append(a_mask)
        v_chunks.append(v_rep)

    model.train()                                  # back to training mode

    a_repr = torch.cat(a_chunks, 0).to(torch.bfloat16).to(device)
    v_repr = torch.cat(v_chunks, 0).to(torch.bfloat16).to(device)
    a_mask = torch.cat(m_chunks, 0).to(device)

    return (
        _detach_repr(a_repr),      # F – requires_grad=True, no history
        _detach_repr(v_repr),      # G
        {"a_mask": a_mask},        # extras
    )


def _loss_on_reprs(model, F, G, extras):
    """
    Same calculation as model.forward() **but using pre-computed
    representations**.  Autograd fills F.grad / G.grad.

    Returns
    -------
      loss_scalar  – detached scalar
      grad_F,grad_G
      outputs_dict – fields expected by visualiser & logging
    """
    clip_sims, token_sims = model.compute_all_similarities_tv(
        F, G, extras["a_mask"]
    )
    loss = model.compute_contrastive_loss_tv(
        clip_sims, token_sims, extras["a_mask"]
    )
    loss.backward()                    # autograd now fills F.grad / G.grad
    outputs = {
        "loss": loss.detach(),
        "clip_sims": clip_sims.detach(),
        "token_sims": token_sims.detach(),
        "audio_feats": F.detach(),
        "visual_feats": G.detach(),
        "audio_attention_mask": extras["a_mask"].detach(),
    }
    return loss.detach(), F.grad.detach(), G.grad.detach(), outputs

# ------------------------------------------------------------------
#  encoder backward with cached grads
# ------------------------------------------------------------------
def _run_audio_backward(model, batch, grad_F, micro_bs, device, use_amp=True):
    """Feeds cached grad_F through audio encoder in micro-batches."""
    audio = batch["audio"].to(device)
    mask  = batch["audio_attention_mask"].to(device)

    for s in range(0, audio.size(0), micro_bs):
        e = min(s + micro_bs, audio.size(0))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            feats, _ = model.audio_embedder(audio[s:e], mask[s:e])
        feats.backward(grad_F[s:e].to(device))


def _run_vision_backward(model, batch, grad_G, micro_bs, device, use_amp=True):
    """Feeds cached grad_G through visual encoder in micro-batches."""
    images = batch["image"].to(device)

    for s in range(0, images.size(0), micro_bs):
        e = min(s + micro_bs, images.size(0))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            feats = model.visual_embedder(images[s:e])
        feats.backward(grad_G[s:e].to(device))



# ---------------------------------------------------------------------------
#  BLOCK-WISE similarity helper  –  no huge (B,B,Na,Nv) tensor on GPU
# ---------------------------------------------------------------------------
def _clip_sims_blockwise(F, G, a_mask, scale, blk=8):
    """
    F : (B, Na, D)  bf16 or fp32
    G : (B, Nv, D)
    a_mask : (B, Na)
    Returns:
        clip_sims   (B,B)  bf16  (cast back to fp32 only if you need high-precision logging)
        token_diag  (B,Na,Nv) bf16
    """
    B, Na, _ = F.shape
    Nv = G.shape[1]
    rows, token_diag = [], []

    # make sure we compute inside an autocast region
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):

        for i0 in range(0, B, blk):
            i1 = min(i0 + blk, B)
            Fi   = F[i0:i1]                     # (bi,Na,D)
            Mi   = a_mask[i0:i1].float()
            Vi   = Mi.sum(dim=1, keepdim=True).clamp_min(1e-5)
            row_chunks = []

            for j0 in range(0, B, blk):
                j1 = min(j0 + blk, B)
                Gj = G[j0:j1]                  # (bj,Nv,D)

                sim = torch.einsum(
                    "b t d, j v d -> b j t v", Fi, Gj
                ) * scale.to(F.dtype)          # bf16 × bf16

                a2v_clip = (sim.max(3).values * Mi.unsqueeze(1)).sum(2) / Vi
                v2a_clip = sim.max(2).values.mean(2)
                row_chunks.append(0.5 * (a2v_clip + v2a_clip))

                if i0 == j0:                   # store diagonal for viz
                    diag = torch.arange(i1 - i0, device=F.device)
                    token_diag.append(sim[diag, diag])

            rows.append(torch.cat(row_chunks, dim=1))

    clip_sims  = torch.cat(rows, dim=0)
    token_diag = torch.cat(token_diag, dim=0)
    return clip_sims, token_diag



class VeS(nn.Module):
    def __init__(
        self, 
        freeze_hubert_initially=True,
    ):
        super().__init__()

        self.visual_embedder = VisionEncoder()  
        self.visual_processor = AutoProcessor.from_pretrained("facebook/dinov2-with-registers-base")

        self.audio_embedder = AudioEmbedder(freeze_hubert_initially=freeze_hubert_initially)
        self.audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10)))
        self.bias = nn.Parameter(torch.tensor(-10.0))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tv_weight = 0.0001

    def compute_similarity_matrix(self, feats1, feats2):
        """
        token-level cosine similarity between feats1 and feats2.
        feats1: (B, N1, D)
        feats2: (B, N2, D)
        Returns sim: (B, N1, N2)
        """ 
        sim = torch.bmm(feats1, feats2.transpose(1, 2))
        return sim * torch.exp(self.logit_scale)


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
        af = audio_feats.unsqueeze(1).expand(-1, B, -1, -1)          # (B, B, Na, D)
        vf = visual_feats.unsqueeze(0).expand(B, -1, -1, -1)        # (B, B, Nv, D)

        # Full token-level sim
        token_sims = torch.matmul(af, vf.transpose(2, 3))           # (B, B, Na, Nv)
        token_sims = token_sims * torch.exp(self.logit_scale)       # scale by learned temp

        # ------------------------------------------------------------
        # 1)  audio → visual • max over patches, mean over tokens
        # ------------------------------------------------------------
        a2v_max  = token_sims.max(dim=3).values                      # (B, B, Na)
        a_mask   = attention_mask.unsqueeze(1).float().expand(-1, B, -1)
        a2v_sum  = (a2v_max * a_mask).sum(dim=2)                     # (B, B)
        valid_a  = a_mask.sum(dim=2).clamp(min=1e-5)  # Slightly larger epsilon for bf16
        a2v_clip = a2v_sum / valid_a                                 # (B, B)

        # ------------------------------------------------------------
        # 2)  visual → audio • max over tokens, mean over patches
        # ------------------------------------------------------------
        v2a_max  = token_sims.max(dim=2).values                      # (B, B, Nv)
        v2a_clip = v2a_max.mean(dim=2)                               # (B, B)

        clip_sims = 0.5 * (a2v_clip + v2a_clip)                      # (B, B)

        return clip_sims, token_sims

    def compute_regularization_losses_tv(
        self,
        token_sims : torch.Tensor,        # (B, B, Na, Nv)
        attn_mask  : torch.Tensor | None  # (B, Na) or None
    ):
        """
        Regularisation =  l_nonneg  +  tv_weight * l_tv

        • l_nonneg  — pushes all (audio-tok, visual-patch) similarities ≥ 0  
        • l_tv      — total-variation smoothing on the *positive* pair
                    (diagonal b → b) along the audio-token axis
        """
        # ----------------------------------------------------------
        # (1) non-negativity pressure  (unchanged)
        # ----------------------------------------------------------
        neg_sims = token_sims.clamp(min=-20.0, max=0.0)
        l_nonneg = neg_sims.pow(2).mean()

        # Early-exit if temporal smoothing is disabled
        if getattr(self, "tv_weight", 0.0) == 0:
            return l_nonneg

        # ----------------------------------------------------------
        # (2) temporal-variation (TV) smoothing on the diagonal
        # ----------------------------------------------------------
        B = token_sims.size(0)
        device = token_sims.device

        # a2v_max[b, bʹ, t] = max over visual patches
        a2v_max   = token_sims.max(dim=3).values            # (B, B, Na)
        pos_trace = a2v_max[torch.arange(B, device=device),
                            torch.arange(B, device=device)]  # (B, Na)

        if attn_mask is not None:
            m_valid   = attn_mask.float().to(device)        # (B, Na)
            neighbour = m_valid[:, 1:] * m_valid[:, :-1]    # (B, Na-1)

            diffs = (pos_trace[:, 1:] - pos_trace[:, :-1]).pow(2)
            l_tv  = (diffs * neighbour).sum() / neighbour.sum().clamp_min(1.0)
        else:
            l_tv = (pos_trace[:, 1:] - pos_trace[:, :-1]).pow(2).mean()

        # ----------------------------------------------------------
        return l_nonneg + self.tv_weight * l_tv


    
    def compute_contrastive_loss_tv(self, clip_sims: torch.Tensor, token_sims: torch.Tensor, attention_mask: torch.Tensor): #sigmoid
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

        #scaling loss for bf16 stability
        #pairwise_loss = pairwise_loss * 10

        # optional regularisation (unchanged from the original implementation)
        #reg_loss   = self.compute_regularization_losses_tv(token_sims, attention_mask)

        total_loss = pairwise_loss# + reg_loss

        return total_loss
    
    def forward(self, audio_input, images, attention_mask=None):
        """
        Forward pass of VeS model.
        
        Args:
            audio_input: (B, T) raw audio waveform at 16kHz
            images: PIL Images or preprocessed image tensors
            
        Returns:
            dict containing:
                - loss: total loss (contrastive + regularization)
                - clip_sims: (B, B) clip-level similarity matrix
                - audio_feats: (B, Na, D) audio features
                - visual_feats: (B, Nv, D) visual features
        """
        audio_feats, attention_mask = self.audio_embedder(audio_input, attention_mask)
        visual_feats = self.visual_embedder(images)
        clip_sims, token_sims = self.compute_all_similarities_tv(
            audio_feats, 
            visual_feats, 
            attention_mask
        )
        loss = self.compute_contrastive_loss_tv(clip_sims, token_sims, attention_mask)
        return {
            'loss': loss,
            'clip_sims': clip_sims.detach(),
            'audio_feats': audio_feats.detach(),
            'visual_feats': visual_feats.detach(),
            'audio_attention_mask': attention_mask.detach(),
            'token_sims': token_sims.detach()
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
'''
if __name__ == "__main__":
    print("Testing VeS with random inputs...")
    
    # Test with staged training (HuBERT frozen initially)
    print("\n=== Testing with HuBERT frozen (Stage 1) ===")
    model = VeS(freeze_hubert_initially=True).to("cuda")

    audio_input, images = dummy_inputs()
    audio_input = audio_input.to("cuda")
    images = images.to("cuda")
    
    # Count trainable params before unfreezing
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (Stage 1): {trainable_before:,}")
    
    outputs = model(audio_input, images)
    print(f"Visual features shape: {outputs['visual_feats'].shape}")
    print(f"Audio features shape: {outputs['audio_feats'].shape}")
    print(f"Clip similarities shape: {outputs['clip_sims'].shape}")
    
    # Test unfreezing
    print("\n=== Testing HuBERT unfreezing (Stage 2) ===")
    model.unfreeze_hubert()
    
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (Stage 2): {trainable_after:,}")
    print(f"Newly unfrozen params: {trainable_after - trainable_before:,}")
    
    model.train()
    with torch.no_grad():
        outputs = model(audio_input, images)
        print(f"Visual features shape: {outputs['visual_feats'].shape}")
        print(f"Audio features shape: {outputs['audio_feats'].shape}")
        print(f"Clip similarities shape: {outputs['clip_sims'].shape}")
'''
import sys

if __name__ == "__main__" and "--test_cache" in sys.argv:
    from model import VeS
    from data import VAAPairedDataset
    import itertools, sys

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = VeS().to(dev).train()

    ds  = VAAPairedDataset()
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))

    # ---- STEP-1 test ----------------------------------------------------
    F, G, ext = _encode_representations(model, batch, dev)
    loss, gF, gG = _loss_on_reprs(model, F, G, ext)

    print(f"cache-only forward OK  – loss {loss.item():.4f}")
    print(f"dL/dF   shape {gF.shape},  mean |grad| {gF.abs().mean():.3e}")
    print(f"dL/dG   shape {gG.shape},  mean |grad| {gG.abs().mean():.3e}")
    sys.exit(0)


if __name__ == "__main__" and "--test_cache2" in sys.argv:
    import sys
    from model import VeS
    from data import VAAPairedDataset
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_micro = 8
    model = VeS().to(dev).train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ds = VAAPairedDataset()
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=12)))

    # ---- cache part (Step 1) --------------------------------------------
    F, G, extra = _encode_representations(model, batch, dev)
    loss, gF, gG = _loss_on_reprs(model, F, G, extra)
    print(f"loss on reprs: {loss.item():.4f}")

    # ---- encoder backward (Step 2) --------------------------------------
    optim.zero_grad(set_to_none=True)

    _run_audio_backward(model, batch, gF, cfg_micro, dev)
    _run_vision_backward(model, batch, gG, cfg_micro, dev)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

    print(f"after step, total grad-norm {grad_norm:.3f}")
    sys.exit(0)
