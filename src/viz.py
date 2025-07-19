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
import matplotlib.colors as mcolors


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
        token_hz: int = 25,                 # 1 / (0.02s * 2) = 25 Hz
        alpha: float = 0.9,                # heat-map opacity
        max_samples_per_call: int = 4,
        reduction: int = 2,
        side_by_side: bool = True,         # Enable side-by-side layout
        separator_width: int = 4,           # Width of separator line between images
        label_height: int = 30,             # Height for text labels
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.fps = token_hz                 # video FPS
        self.alpha = alpha
        self.reduction = reduction
        self.samples_per_token = 320 * reduction
        self.max_samples_per_call = max_samples_per_call
        
        # Side-by-side layout options
        self.side_by_side = side_by_side
        self.separator_width = separator_width
        self.label_height = label_height

        # ImageNet mean / std for un-normalising
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Create inferno colormap with alpha
        self._inferno_cmap = self._get_inferno_with_alpha()

    # ---------------------------------------------------------------------
    #                         internal helpers
    # ---------------------------------------------------------------------

    def _get_inferno_with_alpha(self):
        """Create inferno colormap with alpha channel that varies from transparent to opaque."""
        inferno = plt.cm.inferno(np.linspace(0, 1, 256))
        alphas = np.linspace(0, 1, 256)
        inferno_with_alpha = np.zeros((256, 4))
        inferno_with_alpha[:, 0:3] = inferno[:, 0:3]
        inferno_with_alpha[:, 3] = alphas
        return mcolors.ListedColormap(inferno_with_alpha)

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
        """One token's patch-similarity vector → RGBA heat-map with inferno colormap."""
        #print(sim_row.shape)
        assert sim_row.shape == torch.Size([256]), f"Expected sim_row shape [256], got {sim_row.shape}"
        arr = sim_row.view(grid, grid).float().cpu()
        
        # Normalize similarity values
        arr = arr.clamp_min(0)
        arr = arr - arr.min()
        arr = arr / (arr.max() + 1e-6)
        
        # Resize to target size
        arr_resized = cv2.resize(arr.numpy(), (size, size), interpolation=cv2.INTER_CUBIC)
        
        # Apply inferno colormap with alpha - this creates RGBA values
        heat_rgba = self._inferno_cmap(arr_resized)  # Shape: (size, size, 4)
        
        # Convert to uint8
        heat_rgba = (heat_rgba * 255).astype(np.uint8)
        
        return heat_rgba

    def _blend(self, rgb: np.ndarray, heat_rgba: np.ndarray) -> np.ndarray:
        """Blend RGB image with RGBA heatmap using proper alpha compositing."""
        # Extract RGB and alpha from heatmap
        heat_rgb = heat_rgba[:, :, :3].astype(np.float32) / 255.0
        heat_alpha = heat_rgba[:, :, 3:4].astype(np.float32) / 255.0
        
        # Convert base image to float
        rgb_base = rgb.astype(np.float32) / 255.0
        
        # Alpha blend: result = heat * alpha + base * (1 - alpha)
        alpha_3ch = np.repeat(heat_alpha, 3, axis=2)
        result = heat_rgb * alpha_3ch + rgb_base * (1 - alpha_3ch)
        
        # Convert back to uint8
        return (result * 255).astype(np.uint8)

    def _calculate_crop_region(self, crop_info: dict | None, video_size: int = 448, sample_idx: int = 0) -> tuple[int, int, int, int]:
        """
        Calculate the crop region to remove black padding from pad_square strategy.
        
        Args:
            crop_info: Dictionary with original_width, original_height, crop_strategy, target_size
            video_size: Current video frame size (448x448)
            
        Returns:
            tuple: (x1, y1, x2, y2) crop coordinates
        """
        if crop_info is None:
            # No cropping needed when no crop_info
            return (0, 0, video_size, video_size)
            
        # Handle crop_strategy which might be a batched tensor
        crop_strategy = crop_info["crop_strategy"]
        
        # Handle batched tensors by extracting the value for this sample
        if hasattr(crop_strategy, '__getitem__') and hasattr(crop_strategy, 'shape') and len(crop_strategy.shape) > 0:
            crop_strategy = crop_strategy[sample_idx]
        
        # Convert to scalar/string if it's a tensor
        if hasattr(crop_strategy, 'item'):
            crop_strategy = crop_strategy.item()
        elif isinstance(crop_strategy, bytes):
            crop_strategy = crop_strategy.decode('utf-8')
            
        if crop_strategy != "pad_square":
            # No cropping needed for other strategies
            return (0, 0, video_size, video_size)
        
        # Extract values for this specific sample from potentially batched tensors
        orig_w = crop_info["original_width"]
        orig_h = crop_info["original_height"]
        target_size = crop_info["target_size"]  # Usually 224
        
        # Handle batched tensors by extracting the value for this sample
        if hasattr(orig_w, '__getitem__') and hasattr(orig_w, 'shape') and len(orig_w.shape) > 0:
            orig_w = orig_w[sample_idx]
        if hasattr(orig_h, '__getitem__') and hasattr(orig_h, 'shape') and len(orig_h.shape) > 0:
            orig_h = orig_h[sample_idx]
        if hasattr(target_size, '__getitem__') and hasattr(target_size, 'shape') and len(target_size.shape) > 0:
            target_size = target_size[sample_idx]
            
        # Convert to scalars if they're still tensors
        if hasattr(orig_w, 'item'):
            orig_w = orig_w.item()
        if hasattr(orig_h, 'item'):
            orig_h = orig_h.item()
        if hasattr(target_size, 'item'):
            target_size = target_size.item()
        
        # Calculate what happened during pad_square processing:
        # 1. Original image was padded to max(orig_w, orig_h) square
        # 2. Then resized to target_size (224)
        # 3. Now it's been resized again to video_size (448)
        
        max_dim = max(orig_w, orig_h)
        
        # Calculate the relative size of actual content in the square
        content_w_ratio = orig_w / max_dim
        content_h_ratio = orig_h / max_dim
        
        # Scale to current video size
        content_w_pixels = int(content_w_ratio * video_size)
        content_h_pixels = int(content_h_ratio * video_size)
        
        # Ensure dimensions are divisible by 2 for video encoding
        if content_w_pixels % 2 != 0:
            content_w_pixels -= 1
        if content_h_pixels % 2 != 0:
            content_h_pixels -= 1
        
        # Calculate crop coordinates (centered)
        x1 = (video_size - content_w_pixels) // 2
        y1 = (video_size - content_h_pixels) // 2
        x2 = x1 + content_w_pixels
        y2 = y1 + content_h_pixels
        
        return (x1, y1, x2, y2)

    def _apply_crop(self, frame: np.ndarray, crop_coords: tuple[int, int, int, int]) -> np.ndarray:
        """Apply cropping to a frame and return the cropped region."""
        x1, y1, x2, y2 = crop_coords
        return frame[y1:y2, x1:x2]

    def _create_side_by_side_frame(self, left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
        """
        Create a side-by-side frame with labels and separator.
        
        Args:
            left_frame: Original image frame (H, W, 3)
            right_frame: Heatmap overlay frame (H, W, 3)
            
        Returns:
            Combined frame with labels and separator
        """
        H, W, C = left_frame.shape
        
        # Create canvas with extra height for labels
        total_width = W * 2 + self.separator_width
        total_height = H + self.label_height
        canvas = np.zeros((total_height, total_width, C), dtype=np.uint8)
        
        # Add frames to canvas (below labels)
        canvas[self.label_height:, :W] = left_frame
        canvas[self.label_height:, W + self.separator_width:] = right_frame
        
        # Add separator line
        if self.separator_width > 0:
            separator_color = (128, 128, 128)  # Gray separator
            canvas[self.label_height:, W:W + self.separator_width] = separator_color
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        font_thickness = 2
        
        # "Original" label (left side)
        text_size = cv2.getTextSize("Original", font, font_scale, font_thickness)[0]
        text_x = (W - text_size[0]) // 2
        text_y = self.label_height - 8
        cv2.putText(canvas, "Original", (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        
        # "Heatmap" label (right side)
        text_size = cv2.getTextSize("Heatmap", font, font_scale, font_thickness)[0]
        text_x = W + self.separator_width + (W - text_size[0]) // 2
        cv2.putText(canvas, "Heatmap", (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        
        return canvas

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
        crop_info: dict | None = None,      # New parameter for crop information
        sample_idx: int = 0,                # New parameter for sample index in batch
    ):
        # --- prepare video base frame ------------------------------------
        rgb_base = self._unnormalise(image_t)               # uint8 H×W×3 (224×224)
        # Double the size of the video
        rgb_base = cv2.resize(rgb_base, (448, 448), interpolation=cv2.INTER_CUBIC)
        
        # Calculate crop region to remove black padding
        crop_coords = self._calculate_crop_region(crop_info, video_size=448, sample_idx=sample_idx)
        
        # Apply initial crop to base frame to determine final video dimensions
        rgb_base_cropped = self._apply_crop(rgb_base, crop_coords)
        H, W, _ = rgb_base_cropped.shape

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
        
        # Set video dimensions based on layout mode
        if self.side_by_side:
            vstream.width = W * 2 + self.separator_width
            vstream.height = H + self.label_height
        else:
            vstream.width = W
            vstream.height = H

        astream = container.add_stream("aac", rate=int(sr))
        astream.layout = "mono"

        # --- write video frames ------------------------------------------
        for t in range(valid_tok):
            heat_rgba = self._sim_to_heatmap(token_sims[t], size=448)  # Match doubled video size
            frame_np = self._blend(rgb_base, heat_rgba)
            
            # Apply cropping to remove black padding
            frame_np_cropped = self._apply_crop(frame_np, crop_coords)

            # 🌱 timestamp overlay (bottom-left of heatmap frame)
            ts_sec = t / self.fps
            cv2.putText(
                frame_np_cropped,
                f"{ts_sec:5.2f}s",
                (10, frame_np_cropped.shape[0] - 10),  # Adjust for new height
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Collect frames for matplotlib (clean heatmap overlay only)
            if t in frame_indices:
                collected_frames.append(frame_np_cropped.copy())
                collected_timestamps.append(ts_sec)

            # Create final video frame based on layout mode
            if self.side_by_side:
                # Create original frame (cropped, no heatmap)
                original_cropped = self._apply_crop(rgb_base, crop_coords)
                
                # Create side-by-side frame for video
                video_frame = self._create_side_by_side_frame(original_cropped, frame_np_cropped)
            else:
                # Use the heatmap overlay frame as-is
                video_frame = frame_np_cropped

            frame = av.VideoFrame.from_ndarray(video_frame, format="rgb24")
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
        #fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        
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
        # Handle crop_info first - handle various possible structures from DataLoader
        crop_infos = []
        if "crop_info" in batch:
            raw_crop_infos = batch["crop_info"]
            
            # Handle different possible structures
            if isinstance(raw_crop_infos, list):
                # Expected case: list of crop_info dicts
                crop_infos = raw_crop_infos
            elif isinstance(raw_crop_infos, dict):
                # If DataLoader somehow batched the dict fields
                # Extract each sample's crop_info
                batch_size = len(batch["image_path"]) if "image_path" in batch else len(batch["image"])
                for i in range(batch_size):
                    try:
                        sample_crop_info = {}
                        for k, v in raw_crop_infos.items():
                            if isinstance(v, (list, tuple)):
                                sample_crop_info[k] = v[i]
                            elif hasattr(v, '__getitem__') and hasattr(v, 'shape') and len(v.shape) > 0:
                                # Handle PyTorch tensors and similar array-like objects
                                sample_crop_info[k] = v[i]
                            else:
                                # Scalar value, same for all samples
                                sample_crop_info[k] = v
                        crop_infos.append(sample_crop_info)
                    except (IndexError, KeyError):
                        crop_infos.append(None)
            else:
                # Fallback: fill with None
                batch_size = len(batch["image_path"]) if "image_path" in batch else len(batch["image"])
                crop_infos = [None] * batch_size
        else:
            # Fallback: fill with None if no crop_info
            batch_size = len(batch["image_path"]) if "image_path" in batch else len(batch["image"])
            crop_infos = [None] * batch_size

        if "image" not in batch:  # Using cached features, need to load images
            # Load images from paths for visualization
            image_paths = batch["image_path"]  # List of paths
            
            imgs = []
            for i, img_path in enumerate(image_paths):
                # Load without augmentations for clean viz
                from PIL import Image as PILImage
                import torchvision.transforms as transforms
                
                image = PILImage.open(img_path).convert('RGB')
                
                # Apply the same crop strategy that was used during training
                # (but without augmentations)
                crop_info = crop_infos[i] if i < len(crop_infos) and crop_infos[i] is not None else {}
                
                # Extract and convert values to scalars
                crop_strategy = crop_info.get("crop_strategy", "pad_square")
                target_size = crop_info.get("target_size", 224)
                
                # Convert crop_strategy to string if it's a tensor/bytes
                if hasattr(crop_strategy, 'item'):
                    crop_strategy = crop_strategy.item()
                elif isinstance(crop_strategy, bytes):
                    crop_strategy = crop_strategy.decode('utf-8')
                
                # Convert target_size to int if it's a tensor
                if hasattr(target_size, 'item'):
                    target_size = int(target_size.item())
                else:
                    target_size = int(target_size)
                
                if crop_strategy == "pad_square":
                    width, height = image.size
                    max_dim = max(width, height)
                    new_image = PILImage.new('RGB', (max_dim, max_dim), color=(0, 0, 0))
                    paste_x = (max_dim - width) // 2
                    paste_y = (max_dim - height) // 2
                    new_image.paste(image, (paste_x, paste_y))
                    image = new_image.resize((target_size, target_size), PILImage.LANCZOS)
                # Add other crop strategies as needed...
                
                # Convert to tensor without augmentations
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                
                imgs.append(transform(image))
            
            imgs = torch.stack(imgs).cpu()  # (B, 3, 224, 224)
        else:
            # Images were loaded by dataset
            imgs = batch["image"].cpu()

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
                crop_info=crop_infos[i] if i < len(crop_infos) else None,
                sample_idx=i,
            )
            
            # Create matplotlib figure
            if frames:
                fig = self._create_frame_plot(frames, timestamps, basename)
                if fig is not None:
                    matplotlib_figures.append((basename, fig))
        
        return matplotlib_figures


if __name__ == "__main__":
    import requests
    from PIL import Image
    import argparse
    
    def test_heatmap_visualization():
        """Test the heatmap visualization with a sample image."""
        parser = argparse.ArgumentParser(description="Test heatmap visualization")
        parser.add_argument("--image", type=str, 
                          default="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",
                          help="Image URL or local path")
        parser.add_argument("--output", type=str, default="test_viz", 
                          help="Output directory")
        parser.add_argument("--side-by-side", action="store_true",
                          help="Enable side-by-side layout with original image")
        args = parser.parse_args()
        
        # Create visualizer
        viz = VeSVisualizer(out_dir=args.output, alpha=0.4, side_by_side=args.side_by_side)
        
        # Load image
        if args.image.startswith("http"):
            response = requests.get(args.image)
            img = Image.open(requests.get(args.image, stream=True).raw)
        else:
            img = Image.open(args.image)
        
        # Convert to tensor format (224x224, normalized)
        img = img.convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_array - mean) / std
        img_tensor = torch.tensor(img_norm).permute(2, 0, 1).float()
        
        # Generate fake similarity data with some interesting patterns
        n_tokens = 50  # ~2 seconds at 25 Hz
        sim_data = torch.zeros(n_tokens, 256)
        
        for t in range(n_tokens):
            # Create some moving hotspots and patterns
            peak1_x = int(8 + 6 * np.sin(t * 0.3))  # Moving horizontally
            peak1_y = int(8 + 4 * np.cos(t * 0.2))  # Moving vertically
            peak2_x = int(12 + 3 * np.sin(t * 0.4 + 1))
            peak2_y = int(6 + 3 * np.cos(t * 0.3 + 1))
            
            # Base random similarity
            sim_grid = torch.randn(16, 16) * 0.1 + 0.3
            
            # Add peaks
            sim_grid[peak1_y, peak1_x] = 0.9 + 0.1 * np.sin(t * 0.5)
            sim_grid[peak2_y, peak2_x] = 0.8 + 0.1 * np.cos(t * 0.4)
            
            # Add some spreading around peaks
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if 0 <= peak1_y + dy < 16 and 0 <= peak1_x + dx < 16:
                        sim_grid[peak1_y + dy, peak1_x + dx] += 0.3 * np.exp(-(dx**2 + dy**2) * 0.5)
            
            sim_data[t] = sim_grid.flatten()
        
        # Generate fake audio (silence)
        audio_samples = n_tokens * viz.samples_per_token
        audio = np.zeros(audio_samples, dtype=np.float32)
        
        # Create attention mask (all valid)
        attn_mask = torch.ones(n_tokens)
        
        # Generate the visualization
        print(f"Generating test visualization with {n_tokens} frames...")
        frames, timestamps = viz._encode_sample(
            image_t=img_tensor,
            token_sims=sim_data,
            audio_np=audio,
            sr=16000,
            basename="test_heatmap",
            attn_mask=attn_mask,
            crop_info=None,
        )
        
        # Create matplotlib figure
        if frames:
            fig = viz._create_frame_plot(frames, timestamps, "test_heatmap")
            if fig is not None:
                layout_mode = "side-by-side" if args.side_by_side else "overlay"
                print(f"✅ Test visualization saved to {args.output}/ ({layout_mode} mode)")
                print(f"   Video: test_heatmap.mp4")
                print(f"   Frames: test_heatmap_frames.png")
            else:
                print("❌ Failed to create matplotlib figure")
        else:
            print("❌ No frames collected")
    
    test_heatmap_visualization()
