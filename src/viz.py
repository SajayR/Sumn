# new_visualizer.py
# Completely rewritten "token-sync" visualiser for the new padded-audio
# pipeline (one video-frame every 20 ms = 50 FPS, audio copied verbatim).

import math
import os
from pathlib import Path

import av                                   # pip install av --no-binary av
import cv2                                   # pip install opencv-python-headless
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class VeSVisualizer:
    """
    • Takes token-level similarity rows (Na × Nv) and overlays them as heat-maps
      onto the input RGB frame *in real time* – one frame per HuBERT token
      (20 ms → 50 FPS).
    • Muxes the untouched mono waveform back in, so A/V stays sample-accurate.
    """

    def __init__(
        self,
        out_dir: str | Path = "visualizations",
        token_hz: int = 50,                 # 1 / 0.02 s
        alpha: float = 0.25,                # heat-map opacity
        max_samples_per_call: int = 4,
        reduction: int = 2,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.fps = token_hz                 # video FPS
        self.alpha = alpha
        self.reduction = reduction
        self.samples_per_token = 320 * reduction
        self.max_samples_per_call = max_samples_per_call

        # ImageNet mean / std for un-normalising
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # ---------------------------------------------------------------------
    #                         internal helpers
    # ---------------------------------------------------------------------

    def _unnormalise(self, img_t: torch.Tensor) -> np.ndarray:
        """(3,224,224) float-tensor → uint8 RGB H×W×C."""
        img = img_t.cpu() * self._std + self._mean
        img = (img.clamp(0, 1) * 255).byte()
        return img.permute(1, 2, 0).numpy()

    def _sim_to_heatmap(
        self,
        sim_row: torch.Tensor,              # (Nv,)  (Nv = 16 × 16 = 256)
        grid: int = 16,
        size: int = 224,
    ) -> np.ndarray:
        """One token's patch-similarity vector → coloured uint8 RGB heat-map."""
        #print(sim_row.shape)
        assert sim_row.shape == torch.Size([256]), f"Expected sim_row shape [256], got {sim_row.shape}"
        arr = sim_row.view(grid, grid).float().cpu()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        arr = (arr * 255).byte().numpy()
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_CUBIC)
        heat = cv2.applyColorMap(arr, cv2.COLORMAP_JET)          # BGR
        return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)             # → RGB

    def _blend(self, rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
        return cv2.addWeighted(rgb, 1.0 - self.alpha, heat, self.alpha, 0)

    # ---------------------------------------------------------------------
    #                       encode ONE sample to MP4
    # ---------------------------------------------------------------------

    def _encode_sample(
        self,
        image_t: torch.Tensor,              # (3,224,224)  cpu
        token_sims: torch.Tensor,           # (Na,Nv)      cpu
        audio_np: np.ndarray,               # (T,) float32 / int16
        sr: int,
        basename: str,
        attn_mask: torch.Tensor,            # (Na,) float  cpu
    ):
        # --- prepare video base frame ------------------------------------
        rgb_base = self._unnormalise(image_t)               # uint8 H×W×3 (224×224)
        # Double the size of the video
        rgb_base = cv2.resize(rgb_base, (448, 448), interpolation=cv2.INTER_CUBIC)
        H, W, _ = rgb_base.shape

        # --- figure out "valid" token count ------------------------------
        mask_tok  = int(attn_mask.round().sum().item())      # robust, still an int
        sim_tok   = token_sims.size(0)                       # what we actually have
        #print(f"mask_tok: {mask_tok}, sim_tok: {sim_tok}")
        valid_tok = min(mask_tok, sim_tok)                   # stay in bounds

        token_sims = token_sims[:valid_tok]                  # (valid_tok, Nv)

        # audio: 320 samples per token @16 kHz
        expected_samples = valid_tok * self.samples_per_token
        audio_np = audio_np[:expected_samples]

        # --- collect frames for matplotlib visualization -------------------
        frame_indices = np.linspace(0, valid_tok - 1, min(6, valid_tok), dtype=int)
        collected_frames = []
        collected_timestamps = []

        # --- open container & streams ------------------------------------
        path = self.out_dir / f"{basename}.mp4"
        container = av.open(str(path), mode="w")

        vstream = container.add_stream("libx264", rate=self.fps)
        vstream.pix_fmt = "yuv420p"
        vstream.width = W
        vstream.height = H

        astream = container.add_stream("aac", rate=int(sr))
        astream.layout = "mono"

        # --- write video frames ------------------------------------------
        for t in range(valid_tok):
            heat = self._sim_to_heatmap(token_sims[t], size=448)  # Match doubled video size
            frame_np = self._blend(rgb_base, heat)

            # 🌱 timestamp overlay (bottom-left)
            ts_sec = t * 0.02
            cv2.putText(
                frame_np,
                f"{ts_sec:5.2f}s",
                (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Collect frames for matplotlib
            if t in frame_indices:
                collected_frames.append(frame_np.copy())
                collected_timestamps.append(ts_sec)

            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for pkt in vstream.encode(frame):
                container.mux(pkt)

        # flush any pending frames
        for pkt in vstream.encode():
            container.mux(pkt)

        # --- write audio --------------------------------------------------
        if audio_np.dtype.kind == "f":
            audio16 = (np.clip(audio_np, -1, 1) * 32767).astype(np.int16)
        else:
            audio16 = audio_np.astype(np.int16)

        audio16 = audio16.reshape(1, -1)      # (channels, samples)
        afr = av.AudioFrame.from_ndarray(audio16, format="s16", layout="mono")
        afr.sample_rate = sr

        for pkt in astream.encode(afr):
            container.mux(pkt)
        for pkt in astream.encode():
            container.mux(pkt)

        container.close()

        return collected_frames, collected_timestamps

    def _create_frame_plot(
        self,
        frames: list[np.ndarray],
        timestamps: list[float],
        basename: str,
    ) -> plt.Figure:
        """Create a matplotlib figure showing 6 frames from the video."""
        n_frames = len(frames)
        if n_frames == 0:
            return None
            
        # Create figure with appropriate size
        fig = plt.figure(figsize=(15, 10))
        
        # Use gridspec for better layout control
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.1)
        
        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            ax.imshow(frame)
            ax.set_title(f"t = {ts:.2f}s", fontsize=12)
            ax.axis('off')
        
        fig.suptitle(f"Heatmap Visualization: {basename}", fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save the figure
        plot_path = self.out_dir / f"{basename}_frames.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        return fig

    # ---------------------------------------------------------------------
    #              public API — called from the training loop
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def visualize_batch(self, batch: dict, outputs: dict, step: int):
        """
        batch   – dict coming straight out of DataLoader
                  • expects keys: "image", "audio", "sampling_rate",
                                  "audio_attention_mask"
        outputs – dict coming out of model.forward(); MUST include
                  "token_sims" (B,B,Na,Nv)
        """
        imgs   = batch["image"].cpu()                       # (B,3,224,224)
        audio  = batch["audio"].cpu().numpy()               # (B,T)
        sr     = batch["sampling_rate"]                     # list[int] or int
        attn   = outputs["audio_attention_mask"].cpu()        # (B,Na)
        sims   = outputs["token_sims"].cpu()                # (B,B,Na,Nv)

        # Convert sr to per-sample list
        if isinstance(sr, int):
            sr = [sr] * imgs.size(0)

        # Collect matplotlib figures for wandb logging
        matplotlib_figures = []

        for i in range(min(imgs.size(0), self.max_samples_per_call)):
            basename = f"step{step}_idx{i}"
            
            # Generate video and collect frames
            frames, timestamps = self._encode_sample(
                image_t=imgs[i],
                token_sims=sims[i, i],                      # (Na,Nv)
                audio_np=audio[i],
                sr=sr[i],
                basename=basename,
                attn_mask=attn[i],
            )
            
            # Create matplotlib figure
            if frames:
                fig = self._create_frame_plot(frames, timestamps, basename)
                if fig is not None:
                    matplotlib_figures.append((basename, fig))
        
        return matplotlib_figures
