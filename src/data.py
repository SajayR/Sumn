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

def process_image(image_path, crop_strategy="pad_square", target_size=224):
    """
    Args:
        image_path: Path to image file
        crop_strategy: One of ["stretch", "center_crop", "random_crop", "pad_square"]
        target_size: Target size (224 for ViT)
    
    Returns:
        torch.Tensor: Normalized image tensor [3, 224, 224]
    """
    image = PILImage.open(image_path).convert('RGB')
    
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
        new_image = PILImage.new('RGB', (max_dim, max_dim), (0, 0, 0))
        paste_x = (max_dim - width) // 2
        paste_y = (max_dim - height) // 2
        new_image.paste(image, (paste_x, paste_y))        
        image = new_image.resize((target_size, target_size), PILImage.LANCZOS)
    
    else:
        raise ValueError(f"Unknown crop_strategy: {crop_strategy}")
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  
            std=[0.229, 0.224, 0.225]  
        )
    ])
    
    return transform(image)


class VAAPairedDataset(IterableDataset):
    
    def __init__(self, 
                 completed_audio_path="/bigchungus/CisStuff/VeS/dataset/completed_audio_files.txt",
                 json_mapping_path="/bigchungus/CisStuff/VeS/dataset/vaani_hindi_only.json",
                 data_base_path="/bigchungus/data",
                 crop_strategy="pad_square", 
                 target_size=224,
                 max_audio_duration=5.0,
                 sampling_rate=16000):
        super().__init__()
        
        self.data_base_path = data_base_path
        self.crop_strategy = crop_strategy
        self.target_size = target_size
        self.max_audio_duration = max_audio_duration
        self.sampling_rate = sampling_rate
        
        print("Loading completed audio files list...")
        with open(completed_audio_path, 'r') as f:
            self.completed_audio_files = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(self.completed_audio_files)} completed audio files")
        
        print("Loading JSON mapping...")
        with open(json_mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        # Create a mapping from audioFileName to imageFileName
        self.audio_to_image_map = {}
        for item in mapping_data:
            self.audio_to_image_map[item['audioFileName']] = item['imageFileName']
        print(f"Loaded {len(self.audio_to_image_map)} audio-image mappings")
        
        # Filter completed audio files to only include those with image mappings
        self.valid_audio_files = []
        for audio_file in self.completed_audio_files:
            if audio_file in self.audio_to_image_map:
                self.valid_audio_files.append(audio_file)
        
        print(f"Found {len(self.valid_audio_files)} valid audio files with image mappings")
        
        # Shuffle the list for better training
        random.shuffle(self.valid_audio_files)

    def __len__(self):
        return len(self.valid_audio_files)
    
    def __iter__(self):
        for audio_file in self.valid_audio_files:
            try:
                # Get corresponding image file
                image_file = self.audio_to_image_map[audio_file]
                
                # Construct full paths
                audio_path = os.path.join(self.data_base_path, audio_file)
                image_path = os.path.join(self.data_base_path, image_file)
                
                # Check if files exist
                if not os.path.exists(audio_path):
                    print(f"Warning: Audio file not found: {audio_path}")
                    continue
                if not os.path.exists(image_path):
                    print(f"Warning: Image file not found: {image_path}")
                    continue
                
                # Load and process audio
                audio_array, sr = librosa.load(audio_path, sr=self.sampling_rate)
                
                # Truncate or pad audio to max duration
                max_samples = int(self.max_audio_duration * self.sampling_rate)
                original_length = len(audio_array)
                
                if len(audio_array) > max_samples:
                    # Random crop if longer than max duration
                    start_idx = random.randint(0, len(audio_array) - max_samples)
                    audio_array = audio_array[start_idx:start_idx + max_samples]
                    # All samples are valid after cropping
                    attention_mask = np.ones(max_samples, dtype=np.float32)
                else:
                    # Pad with zeros if shorter than max duration
                    padding = max_samples - len(audio_array)
                    audio_array = np.pad(audio_array, (0, padding), mode='constant')
                    # Create attention mask: 1 for real audio, 0 for padding
                    attention_mask = np.concatenate([
                        np.ones(original_length, dtype=np.float32),
                        np.zeros(padding, dtype=np.float32)
                    ])
                
                # Process image
                image_tensor = process_image(image_path, self.crop_strategy, self.target_size)
                
                # Convert audio and mask to tensors
                audio_tensor = torch.from_numpy(audio_array).float()
                attention_mask_tensor = torch.from_numpy(attention_mask).float()
                
                yield {
                    "audio": audio_tensor,  # torch.Tensor [T]
                    "audio_attention_mask": attention_mask_tensor,  # torch.Tensor [T]
                    "sampling_rate": self.sampling_rate,  # int
                    "image": image_tensor,  # torch.Tensor [3, 224, 224]
                    "audio_path": audio_path,
                    "image_path": image_path,
                }
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue

