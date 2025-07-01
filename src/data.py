#### data.py ####
import argparse
from io import BytesIO
import os
import pickle
from functools import partial
import numpy as np
import json
import librosa
from tqdm import tqdm
import torch
from torch.utils.data import IterableDataset
from PIL import Image as PILImage
import random
import torchvision.transforms as transforms
import torchaudio
import torchaudio.transforms as T
from transformers import AutoProcessor
from pathlib import Path

def process_image(image_path, crop_strategy="pad_square", target_size=224, use_augmentations=True):
    """
    Args:
        image_path: Path to image file
        crop_strategy: One of ["stretch", "center_crop", "random_crop", "pad_square"]
        target_size: Target size (224 for ViT)
        use_augmentations: Whether to apply data augmentations (set False when using cached features)
    
    Returns:
        tuple: (torch.Tensor, dict) - Normalized image tensor [3, 224, 224] and original dimensions info
    """
    image = PILImage.open(image_path).convert('RGB')
    original_width, original_height = image.size
    
    if crop_strategy == "stretch":
        # stretch to target size
        image = image.resize((target_size, target_size), PILImage.LANCZOS)
        
    elif crop_strategy == "center_crop":
        #Center crop to square, then resize
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        image = image.crop((left, top, right, bottom))
        image = image.resize((target_size, target_size), PILImage.LANCZOS)
        
    elif crop_strategy == "random_crop":
        # Random crop to square, then resize
        width, height = image.size
        min_dim = min(width, height)
        max_left = width - min_dim
        max_top = height - min_dim
        left = random.randint(0, max_left) if max_left > 0 else 0
        top = random.randint(0, max_top) if max_top > 0 else 0
        right = left + min_dim
        bottom = top + min_dim
        image = image.crop((left, top, right, bottom))
        image = image.resize((target_size, target_size), PILImage.LANCZOS)
        
    elif crop_strategy == "pad_square":
        # Pad to square with black, then resize
        width, height = image.size
        max_dim = max(width, height)
        new_image = PILImage.new('RGB', (max_dim, max_dim), color=(0, 0, 0))
        paste_x = (max_dim - width) // 2
        paste_y = (max_dim - height) // 2
        new_image.paste(image, (paste_x, paste_y))        
        image = new_image.resize((target_size, target_size), PILImage.LANCZOS)
    
    else:
        raise ValueError(f"Unknown crop_strategy: {crop_strategy}")
    
    # Choose transform based on whether we want augmentations
    if use_augmentations:
        # Training transforms with augmentations
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomRotation(degrees=10),
           # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Deterministic transforms (consistent with cache)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    # Create crop info for later use
    crop_info = {
        "original_width": original_width,
        "original_height": original_height,
        "crop_strategy": crop_strategy,
        "target_size": target_size,
        "augmentations_used": use_augmentations
    }
    
    return transform(image), crop_info


class VAAPairedDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                 json_dir_path="/speedy/CisStuff/VeS/vaani_data",  # Now it's a directory!
                 data_base_path="/speedy/Vaani",
                 crop_strategy="pad_square", 
                 target_size=224,
                 max_audio_duration=5.0,
                 sampling_rate=16000,
                 debug=False,
                 cached_features_base_path=None,
                 is_validation=False):  # Add this to know which JSONs to load
        super().__init__()
        
        self.data_base_path = data_base_path
        self.crop_strategy = crop_strategy
        self.target_size = target_size
        self.max_audio_duration = max_audio_duration
        self.sampling_rate = sampling_rate
        self.resampler = None
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        
        # Cached features setup
        self.cached_features_base_path = cached_features_base_path
        if cached_features_base_path is not None:
            print(f"Using cached visual features from: {cached_features_base_path}")
            cache_path = Path(cached_features_base_path)
            if cache_path.exists():
                pt_files = list(cache_path.rglob("*.pt"))
                print(f"Found {len(pt_files)} cached .pt files in directory")
        
        # Load all the mappings from JSONs
        json_dir = Path(json_dir_path)
        
        if is_validation:
            # Load the single validation JSON
            val_path = json_dir / "validation_set.json"
            json_files = [val_path] if val_path.exists() else []
        else:
            # Load all train_*.json files
            json_files = sorted(json_dir.glob("train_*.json"))
        
        print(f"Found {len(json_files)} JSON files to load")
        
        # Just load everything into a flat list!
        self.all_mappings = []
        
        for json_path in tqdm(json_files, desc="Loading JSON mappings"):
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.all_mappings.extend(data)
                print(f"  {json_path.name}: {len(data)} entries")
        
        print(f"Total mappings loaded: {len(self.all_mappings)}")
        
        # Audio backend setup
        if 'sox_io' in torchaudio.list_audio_backends():
            torchaudio.set_audio_backend('sox_io')
        elif 'soundfile' in torchaudio.list_audio_backends():
            torchaudio.set_audio_backend('soundfile')

    def __len__(self):
        return len(self.all_mappings)
    
    def __getitem__(self, idx):
        # Direct access to the mapping - so clean!
        mapping = self.all_mappings[idx]
        audio_file = mapping['audioFileName'].lstrip('/')
        image_file = mapping['imageFileName'].lstrip('/')
        
        try:
            # Construct full paths
            audio_path = Path(self.data_base_path) / audio_file
            image_path = Path(self.data_base_path) / image_file
            
            # The files should exist since you validated them, but just in case...
            if not audio_path.exists() or not image_path.exists():
                print(f"Warning: Missing files for idx {idx}")
                return self.__getitem__((idx + 1) % len(self))
            
            # Load and process audio (same as before)
            waveform, sr = torchaudio.load(str(audio_path))
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sr != self.sampling_rate:
                if self.resampler is None or self.resampler.orig_freq != sr:
                    self.resampler = T.Resample(sr, self.sampling_rate)
                waveform = self.resampler(waveform)
            
            audio_tensor = waveform.squeeze(0)
            
            max_samples = int(self.max_audio_duration * self.sampling_rate)
            
            if audio_tensor.shape[0] > max_samples:
                start_idx = random.randint(0, audio_tensor.shape[0] - max_samples)
                audio_tensor = audio_tensor[start_idx:start_idx + max_samples]

            processed = self.processor(
                audio_tensor.numpy(),
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding="max_length",
                max_length=max_samples,
                truncation=True,
                return_attention_mask=True,
            )
            
            audio_tensor = processed.input_values.squeeze(0)
            attention_mask = processed.attention_mask.squeeze(0)
            
            # Check for cached visual features
            cached_features = None
            use_cached = False
            
            if self.cached_features_base_path is not None:
                cache_base = Path(self.cached_features_base_path)
                pt_filename = image_file.replace('.jpg', '.pt').replace('.jpeg', '.pt').replace('.png', '.pt')
                pt_path = cache_base / pt_filename
                
                if pt_path.exists():
                    try:
                        cached_features = torch.load(pt_path, map_location='cpu')
                        use_cached = True
                    except Exception as e:
                        print(f"Warning: Failed to load cached features from {pt_path}: {e}")
                        use_cached = False
            
            # Process image
            use_augmentations = not use_cached
            image_tensor, crop_info = process_image(str(image_path), self.crop_strategy, 
                                                    self.target_size, use_augmentations)
            
            result = {
                "audio": audio_tensor,
                "audio_attention_mask": attention_mask,
                "sampling_rate": self.sampling_rate,
                "image": image_tensor,
                "audio_path": str(audio_path),
                "image_path": str(image_path),
                "crop_info": crop_info,
                "using_cached_features": use_cached
            }
            
            if use_cached and cached_features is not None:
                result["cached_visual_features"] = cached_features
            
            return result
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            return self.__getitem__((idx + 1) % len(self))