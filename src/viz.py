# visualizer.py
# -----------------------------------------------------------
import math, os
from pathlib import Path

import av                           # pip install av --no-binary av  (needs FFmpeg dev libs)
import cv2                          # pip install opencv-python-headless
import numpy as np
import torch


class VeSVisualizer:
    """
    Dumps a *.mp4* per sample showing:
        • original 224×224 RGB image
        • patch–wise heat-map for each chosen audio token
        • original 16-kHz mono waveform as the audio track
    Uses PyAV (FFmpeg) for tight A/V muxing.
    """

    def __init__(
        self,
        out_dir: str | Path = "visualizations",
        fps: int = 25,
        alpha: float = 0.25,                 # heat-map opacity
        max_samples_per_call: int = 4,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.alpha = alpha
        self.max_samples_per_call = max_samples_per_call

        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # ------------------------------------------------------------------
    #                ---- low-level helpers ----
    # ------------------------------------------------------------------

    def _unnormalise(self, img_t: torch.Tensor) -> np.ndarray:
        """Undo ImageNet norm → uint8 RGB (H,W,C)."""
        img = img_t.cpu() * self._std + self._mean
        img = (img.clamp(0, 1) * 255).byte()
        return img.permute(1, 2, 0).numpy()            # RGB

    def _sim_to_heatmap(self, sim_row: torch.Tensor,
                        grid: int = 16,
                        size: int = 224) -> np.ndarray:
        """Nv-vector → coloured (RGB) 224×224 heat-map."""
        arr = sim_row.view(grid, grid).float().cpu()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        arr = (arr * 255).byte().numpy()
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_CUBIC)
        heat = cv2.applyColorMap(arr, cv2.COLORMAP_JET)          # BGR
        return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)             # → RGB

    def _blend(self, rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
        return cv2.addWeighted(rgb, 1.0 - self.alpha, heat, self.alpha, 0)

    def _pick_tokens(self, valid_tokens: int, duration_s: float) -> list[int]:
        """Sub-sample tokens to hit ≈ self.fps frames/sec."""
        target = max(1, round(duration_s * self.fps))
        stride = max(1, math.ceil(valid_tokens / target))
        return list(range(0, valid_tokens, stride))

    # ------------------------------------------------------------------
    #              ---- encode ONE sample to an MP4 ----
    # ------------------------------------------------------------------

    def _encode_sample(
        self,
        image_t: torch.Tensor,           # (3,224,224)  cpu tensor
        token_sims: torch.Tensor,        # (Na,Nv)      cpu tensor
        audio_np: np.ndarray,            # (T,)         float ∈ [-1,1] or int16
        sr: int,
        basename: str,
        attn_mask: torch.Tensor,         # (Na,)        cpu tensor
    ):
        rgb_base = self._unnormalise(image_t)               # uint8 RGB
        
        # Double the size - hardcoded 2x scaling
        original_H, original_W, _ = rgb_base.shape
        rgb_base = cv2.resize(rgb_base, (original_W * 2, original_H * 2), interpolation=cv2.INTER_CUBIC)
        H, W, _ = rgb_base.shape

        valid_tok = int(attn_mask.sum().item())
        duration = len(audio_np) / sr
        tok_ids = self._pick_tokens(valid_tok, duration)
        actual_fps = len(tok_ids) / duration if duration else self.fps

        path = self.out_dir / f"{basename}.mp4"
        container = av.open(str(path), mode="w")

        vstream = container.add_stream("libx264", rate=int(round(actual_fps)))
        vstream.width, vstream.height, vstream.pix_fmt = W, H, "yuv420p"
        astream = container.add_stream("aac", rate=sr)
        astream.layout= "mono"

        # ---- video frames ----
        for t in tok_ids:
            # Generate heatmap at double size
            heat = self._sim_to_heatmap(token_sims[t], size=H)  # Use doubled height as size
            frame_np = self._blend(rgb_base, heat)

            #   (optional) tiny timestamp bottom-left - scale text size for larger video
            ts = t / valid_tok * duration
            cv2.putText(
                frame_np, f"{ts:5.2f}s",
                (10, H - 16), cv2.FONT_HERSHEY_SIMPLEX,  # Scale position and size
                0.8, (255, 255, 255), 2, cv2.LINE_AA,    # Scale font size and thickness
            )
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for pkt in vstream.encode(frame):
                container.mux(pkt)

        # flush pending
        for pkt in vstream.encode():
            container.mux(pkt)

        # ---- audio ----
        if audio_np.dtype.kind == "f":               # float32/64 → int16
            audio16 = (np.clip(audio_np, -1, 1) * 32767).astype(np.int16)
        else:
            audio16 = audio_np.astype(np.int16)
        
        # Reshape to (n_samples, n_channels) for mono audio
        audio16 = audio16.reshape(1, -1)
        
        afr = av.AudioFrame.from_ndarray(audio16, format="s16", layout="mono")
        afr.sample_rate = sr
        for pkt in astream.encode(afr):
            container.mux(pkt)
        for pkt in astream.encode():
            container.mux(pkt)

        container.close()

    # ------------------------------------------------------------------
    #       ---- public entry-point, called from the trainer ----
    # ------------------------------------------------------------------

    @torch.no_grad()
    def visualize_batch(self, batch: dict, outputs: dict, step: int):
        """
        batch   – original batch dict coming out of DataLoader
        outputs – forward() return dict (MUST include token_sims)
        """
        imgs = batch["image"].cpu()
        raw_audio = batch["raw_audio"]
        sr_list = batch["sampling_rate"]
        attn = outputs["audio_attention_mask"].cpu()
        sims = outputs["token_sims"].cpu()          # (B,B,Na,Nv)

        B = imgs.size(0)
        for i in range(min(B, self.max_samples_per_call)):
            self._encode_sample(
                image_t=imgs[i],
                token_sims=sims[i, i],              # diagonal slice
                audio_np=raw_audio[i],
                sr=sr_list[i],
                basename=f"step{step}_idx{i}",
                attn_mask=attn[i],
            )
