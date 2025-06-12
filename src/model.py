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
    
    This version properly handles the attention mask downsampling that occurs
    in HuBERT's convolutional feature extraction layers.
    """
    def __init__(self, embedding_dim=256, hubert_name="ntu-spml/distilhubert"):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(hubert_name)

        self.projection1 = nn.Linear(self.hubert.config.hidden_size, 256)
        self.layer_norm = nn.LayerNorm(256)
        self.projection2 = nn.Linear(256, embedding_dim)
        self.downsample_factor = self._compute_downsample_factor()
        #print(f"Downsample factor: {self.downsample_factor}")
        
        for param in self.hubert.parameters():
            param.requires_grad = True
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True
    
    def _compute_downsample_factor(self):
        """
        Compute the total downsampling factor of HuBERT's feature extractor.
        This is the product of all stride values in the convolutional layers.
        """
        downsample_factor = 1
        if hasattr(self.hubert, 'feature_extractor'):
            for layer in self.hubert.feature_extractor.conv_layers:
                if hasattr(layer, 'conv'):
                    downsample_factor *= layer.conv.stride[0]
        else:
            # Default factor for most HuBERT models
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
        # Get HuBERT outputs
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
            # If no input mask, all outputs are valid
            batch_size = hubert_output.size(0)
            output_attention_mask = torch.ones(
                batch_size, output_length,
                dtype=torch.long,
                device=hubert_output.device
            )
        
        # Project features
        audio_feats = self.projection2(self.layer_norm(self.projection1(hubert_output)))
        audio_feats = F.normalize(audio_feats, dim=-1)
        return audio_feats, output_attention_mask
    
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
        print("Dino input shape: ", x.shape)
        outputs = self.model(pixel_values=x,return_dict=True,output_attentions=False, output_hidden_states=False)
        patches = outputs.last_hidden_state[:,5:, :] #5 cause 1 is the cls token, 4 are registers
        feats = self.projection2(self.layer_norm(self.projection1(patches)))
        if self.training:
            feats = self.patch_dropout(feats, self.patch_dropout_rate)
        feats = F.normalize(feats, dim=-1)
        print("Dino output shape: ", feats.shape)
        return feats



class VeS(nn.Module):
    def __init__(
        self, 
        use_amp=True,
    ):
        super().__init__()

        self.visual_embedder = VisionEncoder()  
        self.visual_processor = AutoProcessor.from_pretrained("facebook/dinov2-with-registers-base")

        self.audio_embedder = AudioEmbedder()
        self.audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10)))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.use_amp = use_amp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # Convert batched tensor to list of 1D arrays for the processor
        if len(audio_input.shape) == 2:  # (B, T)
            audio_list = [audio_input[i].cpu().numpy() for i in range(audio_input.shape[0])]
        else:  # Handle other cases
            audio_list = audio_input.cpu().numpy()
        
        processed = self.audio_processor(
            audio_list, 
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

        if isinstance(images, torch.Tensor):
            # Move to correct device and dtype for raw tensor inputs
            device = next(self.parameters()).device
            pixel_values = images.to(device)
        else:
            pixel_values = self.process_image(images)

        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                audio_feats, attention_mask = self.audio_embedder(processed_audio, audio_attention_mask)
                visual_feats = self.visual_embedder(pixel_values)
                clip_sims, token_sims = self.compute_all_similarities_tv(
                    audio_feats, 
                    visual_feats, 
                    attention_mask
                )
                loss = self.compute_contrastive_loss_tv(clip_sims, token_sims)
        else:
            audio_feats, attention_mask = self.audio_embedder(processed_audio, audio_attention_mask)
            visual_feats = self.visual_embedder(pixel_values)
            clip_sims, token_sims = self.compute_all_similarities_tv(
                audio_feats, 
                visual_feats, 
                attention_mask
            )
            loss = self.compute_contrastive_loss_tv(clip_sims, token_sims)
        
        return {
            'loss': loss,
            'clip_sims': clip_sims,
            'audio_feats': audio_feats,
            'visual_feats': visual_feats,
            'audio_attention_mask': attention_mask
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

    
    def compute_contrastive_loss_tv(self, clip_sims: torch.Tensor, token_sims: torch.Tensor): #sigmoid
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

        labels = torch.full_like(clip_sims, -1.0)
        labels.fill_diagonal_(1.0)           # positives on the main diagonal
        logits        = clip_sims + self.bias      # broadcast learnable bias b
        pairwise_loss = -F.logsigmoid(labels * logits).mean()

        #scaling loss for bf16 stability
        #pairwise_loss = pairwise_loss * 10

        # optional regularisation (unchanged from the original implementation)
        reg_loss   = self.compute_regularization_losses_tv(token_sims)

        total_loss = pairwise_loss + reg_loss

        return total_loss


def dummy_inputs():
    """Fixture to create dummy audio and image inputs."""
    batch_size = 2
    audio_seq_len = 16000 * 5  # 5 seconds of audio at 16kHz
    
    # Dummy audio: (B, T)
    audio_input = torch.randn(batch_size, audio_seq_len)
    
    # Dummy image: list of PIL Images
    #images = [Image.new('RGB', (224, 224)) for _ in range(batch_size)]
    images = torch.randn(batch_size, 3, 224, 224)
    
    return audio_input, images
if __name__ == "__main__":
    print("Testing VeS with random inputs...")
    model = VeS(
        use_amp=True,
    ).to("cuda")

    audio_input, images = dummy_inputs()
    # Move dummy inputs to CUDA since model is on CUDA
    audio_input = audio_input.to("cuda")
    images = images.to("cuda")
    
    outputs = model(audio_input, images)
    print(outputs['visual_feats'].shape)
    print(outputs['audio_feats'].shape)
    print(outputs['clip_sims'].shape)
    model.train()
    with torch.no_grad():
        outputs = model(audio_input, images)
        print(outputs['visual_feats'].shape)
        print(outputs['audio_feats'].shape)
        print(outputs['clip_sims'].shape)
