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
import torchvision.transforms as transforms
from PIL import Image
from torch.cuda.amp import autocast
from peft import (
    LoraConfig, 
    get_peft_model,
    TaskType,
)

class AudioEmbedder(nn.Module):
    """
    Pre-trained HuBERT to extract audio features from raw audio (16kHz).
    Projects them down to a desired embedding dimension.
    """
    def __init__(self, embedding_dim=512, hubert_name="ntu-spml/distilhubert"):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(hubert_name)

        self.projection1 = nn.Linear(self.hubert.config.hidden_size, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.projection2 = nn.Linear(512, embedding_dim)
        
        for param in self.hubert.parameters():
            param.requires_grad = True
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True
        
    def forward(self, audio_input: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            audio_input: (B, L) processed audio features
            attention_mask: (B, L) attention mask for audio tokens
            
        Returns:
            audio_feats: (B, Na, D) 
                B = batch size
                Na = number of audio tokens
                D = embedding_dim
        """
        hubert_output = self.hubert(audio_input, attention_mask=attention_mask).last_hidden_state
        audio_feats = self.projection2(self.layer_norm(self.projection1(hubert_output)))
        audio_feats = F.normalize(audio_feats, dim=-1)
        return audio_feats
    
class VisionEncoder(nn.Module):

    def __init__(self, embedding_dim=256, patch_dropout_prob=0.1, lora_rank=16, lora_alpha=32):
        super().__init__()

        self.model = AutoModel.from_pretrained('facebook/dinov2-with-registers-base')
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
            lora_dropout=0.1,
            fan_in_fan_out=True,  
            bias="none",          
            modules_to_save=None,  
            
        )

        self.model = get_peft_model(self.model, lora_config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"VisionEncoder - Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")
        self.projection1 = nn.Linear(self.model.config.hidden_size, 256)
        self.layer_norm = nn.LayerNorm(256)
        self.projection2 = nn.Linear(256, embedding_dim)
        self.patch_dropout_rate = patch_dropout_prob
        self.patch_dropout = self.patch_dropout_layer

        for name, param in self.model.named_parameters():
            if 'lora_' not in name: 
                param.requires_grad = False
            else:
                param.requires_grad = True
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True

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
        
        outputs = self.model(pixel_values=x,return_dict=True,output_attentions=False, output_hidden_states=False)
        patches = outputs.last_hidden_state[:,1:, :]
        feats = self.projection2(self.layer_norm(self.projection1(patches)))
        if self.training:
            feats = self.patch_dropout(feats, self.patch_dropout_rate)
        feats = F.normalize(feats, dim=-1)
        return feats



class VeS(nn.Module):
    def __init__(
        self, 
        use_amp=True,
    ):
        super().__init__()

        self.visual_embedder = VisionEncoder()  
        self.visual_processor = AutoProcessor.from_pretrained("facebook/dinov2-with-registers-base")
        self.audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

        self.audio_embedder = AudioEmbedder()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16
        
    def process_audio(self, audio_input: torch.Tensor):
        """
        Process raw audio waveform using HuBERT processor.
        
        Args:
            audio_input: (B, T) raw audio waveform at 16kHz
            
        Returns:
            processed_audio: processed audio tensor
            attention_mask: (B, L) attention mask for audio tokens
        """
        if len(audio_input.shape) == 3:
            audio_input = audio_input.squeeze(0)
            
        processed = self.audio_processor(
            audio_input, 
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True
        )
        
        device = next(self.parameters()).device
        processed_audio = processed.input_values.to(device)
        attention_mask = processed.attention_mask.to(device)
        
        return processed_audio, attention_mask
    
    def process_image(self, images):
        """
        Process images using DINOv2 processor.
        
        Args:
            images: PIL Image(s) or image tensors
            
        Returns:
            pixel_values: (B, 3, H, W) processed image tensor
        """
        processed = self.visual_processor(images=images, return_tensors="pt")
        device = next(self.parameters()).device
        pixel_values = processed.pixel_values.to(device)
        
        return pixel_values

    def forward(self, audio_input, images):
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
        # Process raw inputs
        processed_audio, audio_attention_mask = self.process_audio(audio_input)
        pixel_values = self.process_image(images)

        if self.use_amp:
            with autocast(dtype=self.amp_dtype):
                audio_feats = self.audio_embedder(processed_audio, audio_attention_mask)
                visual_feats = self.visual_embedder(pixel_values)
                clip_sims, token_sims = self.compute_all_similarities_tv(
                    audio_feats, 
                    visual_feats, 
                    audio_attention_mask
                )
                loss = self.compute_contrastive_loss_tv(clip_sims, token_sims)
        else:
            audio_feats = self.audio_embedder(processed_audio, audio_attention_mask)
            visual_feats = self.visual_embedder(pixel_values)
            clip_sims, token_sims = self.compute_all_similarities_tv(
                audio_feats, 
                visual_feats, 
                audio_attention_mask
            )
            loss = self.compute_contrastive_loss_tv(clip_sims, token_sims)
        
        return {
            'loss': loss,
            'clip_sims': clip_sims,
            'audio_feats': audio_feats,
            'visual_feats': visual_feats,
            'audio_attention_mask': audio_attention_mask
        }

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

        # Broadcast so we can compare every text in the batch with every image

        af = audio_feats.unsqueeze(1).expand(-1, B, -1, -1)          # (B, B, Na, D)
        vf = visual_feats.unsqueeze(0).expand(B, -1, -1, -1)        # (B, B, Nv, D)

        # Full token-level similarity tensor
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

    def compute_regularization_losses_tv(self, token_sims):
 
        # (B, B, Nt, Nv)
        B = token_sims.shape[0]
        # negative clamp
        neg_sims = torch.clamp(token_sims, min=-20, max=0)
        l_nonneg = torch.mean(neg_sims**2)
        return l_nonneg

        
    def compute_contrastive_loss_tv(self, clip_sims, token_sims): #infonce
        
        B = clip_sims.shape[0]
        labels = torch.arange(B, device=clip_sims.device)
        
        # audio->visual
        log_prob_a2v = F.log_softmax(clip_sims, dim=1)
        losses_a2v = -log_prob_a2v[torch.arange(B), labels]

        # visual->audio
        log_prob_v2a = F.log_softmax(clip_sims.t(), dim=1)
        losses_v2a = -log_prob_v2a[torch.arange(B), labels]

        contrastive_loss = (losses_a2v + losses_v2a).mean() / 2
        reg_loss = self.compute_regularization_losses_tv(token_sims)

        total_loss = contrastive_loss + reg_loss
        
        return total_loss



if __name__ == "__main__":
    print("Testing VeS with random inputs...")
    model = VeS(
        use_amp=True,
    ).to("cuda")
    