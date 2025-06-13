# --------------------------------- visualizer.py ---------------------------------
"""
Light-weight batch visualizer for VeS training.

Generates one MP4 per sample in a batch:
  • background  : original 224×224 RGB image  
  • overlay     : heat-map of token-to-patch similarity  
  • audio track : original waveform (16 kHz)

Author: your friendly future self 🙃
"""

from pathlib import Path
import numpy as np
import cv2
import torch
import math
from moviepy.editor import *


# ---------------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------------
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denorm(img_t: torch.Tensor) -> np.ndarray:
    """
    Undo ImageNet norm and convert to uint8 HWC.
    img_t : (3, 224, 224)  torch.Tensor on *any* device
    """
    img = (img_t.cpu() * _STD + _MEAN).clamp(0, 1)        # 0–1
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def _token_sims(audio_feats: torch.Tensor,
                visual_feats: torch.Tensor,
                logit_scale: torch.Tensor) -> torch.Tensor:
    """
    Compute (Na, Nv) sim matrix on CPU,  *no* grad.
    All feature vectors are already L2-normalised in the model.
    """
    with torch.no_grad():
        s = audio_feats @ visual_feats.T            # (Na, Nv)
        s = s * logit_scale.exp()
        return s.cpu()                              # keep CPU throughout


def _make_frames(img_np: np.ndarray,
                 sims: torch.Tensor,                # (Na, Nv) on CPU
                 attn_mask: torch.Tensor,           # (Na,)
                 target_fps: int) -> list[np.ndarray]:
    """
    Produce a list of RGB frames (~224²) at the chosen fps.

    Stride tokens ⇒ nearest to  target_fps  over whole audio duration.
    """
    Na = sims.size(0)
    valid_idx = torch.where(attn_mask > 0)[0]
    if len(valid_idx) == 0:
        return []                                  # nothing real in this example

    # --- choose which tokens become frames ---------------------------------
    audio_dur_sec = len(valid_idx) * 0.02          # 20 ms per token (HuBERT 16 k stride 320)
    stride = max(1, round(len(valid_idx) / (audio_dur_sec * target_fps)))
    chosen = valid_idx[::stride]
    fps    = len(chosen) / audio_dur_sec

    # --- pre-compute the coloured heat-maps ---------------------------------
    frames = []
    for t in chosen:
        h = sims[t]                                # (Nv,)
        h = (h - h.min()) / (h.max() - h.min() + 1e-6)  # 0–1
        h = h.view(14, 14).numpy().astype(np.float32)
        h = cv2.resize(h, (224, 224), interpolation=cv2.INTER_LINEAR)
        h = (h * 255).astype(np.uint8)
        h = cv2.applyColorMap(h, cv2.COLORMAP_JET)         # BGR uint8

        # alpha-blend with background
        frame = cv2.addWeighted(img_np, 0.6, h, 0.4, 0)    # still BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    return frames, fps


# ---------------------------------------------------------------------------------
#  public API
# ---------------------------------------------------------------------------------
def visualise_batch(batch_idx: int,
                    images: torch.Tensor,                  # (B,3,224,224)  still *normalized*
                    audio_feats: torch.Tensor,             # (B,Na,D)  GPU ok
                    visual_feats: torch.Tensor,            # (B,Nv,D)
                    audio_attention_mask: torch.Tensor,    # (B,Na)
                    logit_scale: torch.Tensor,             # scalar param
                    raw_audio: list[np.ndarray],           # len=B lists
                    out_dir: Path,
                    fps: int = 25):
    """
    Creates MP4s under  out_dir/batch{batch_idx}_sample{i}.mp4
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    B = images.size(0)
    logit_scale_cpu = logit_scale.detach().cpu()

    for i in range(B):
        img_np = _denorm(images[i])
        sims   = _token_sims(audio_feats[i].detach().cpu(),
                             visual_feats[i].detach().cpu(),
                             logit_scale_cpu)

        frames, true_fps = _make_frames(img_np, sims, audio_attention_mask[i].cpu(), fps)
        if not frames:          # nothing to draw
            continue

        clip = ImageSequenceClip(frames, fps=true_fps)
        # moviepy expects [-1,1] float or int16 audio
        wav = raw_audio[i]
        # ensure float32 -1..1
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32) / np.abs(wav).max()
        audio_clip = AudioArrayClip(wav, fps=16000)
        clip = clip.set_audio(audio_clip)

        fname = out_dir / f"batch{batch_idx}_sample{i}.mp4"
        clip.write_videofile(fname.as_posix(),
                             codec="libx264",
                             audio_codec="aac",
                             verbose=False,
                             logger=None)          # suppress moviepy spam
