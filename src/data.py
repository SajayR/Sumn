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

def process_image(image_path, crop_strategy="pad_square", target_size=224):
    """
    Args:
        image_path: Path to image file
        crop_strategy: One of ["stretch", "center_crop", "random_crop", "pad_square"]
        target_size: Target size (224 for ViT)
    
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
    

    '''transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  
            std=[0.229, 0.224, 0.225]  
        )
    ])'''

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
    
    # Create crop info for later use
    crop_info = {
        "original_width": original_width,
        "original_height": original_height,
        "crop_strategy": crop_strategy,
        "target_size": target_size
    }
    
    return transform(image), crop_info


# Change from IterableDataset to Dataset
class VAAPairedDataset(torch.utils.data.Dataset):  # ← Change this
    
    def __init__(self, 
                 completed_audio_path="/speedy/CisStuff/dataset/completed_audio_files_disk.txt",
                 json_mapping_path="/speedy/CisStuff/dataset/hindi_filtered_dataset.json",
                 data_base_path="/speedy/Vaani",
                 crop_strategy="pad_square", 
                 target_size=224,
                 max_audio_duration=5.0,
                 sampling_rate=16000,
                 debug=False):
        super().__init__()
        
        self.data_base_path = data_base_path
        self.crop_strategy = crop_strategy
        self.target_size = target_size
        self.max_audio_duration = max_audio_duration
        self.sampling_rate = sampling_rate
        self.resampler = None  # Initialize resampler as None
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        
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

        # In __init__:
        # Check available backends
        if 'sox_io' in torchaudio.list_audio_backends():
            torchaudio.set_audio_backend('sox_io')  # Fast for many formats
        elif 'soundfile' in torchaudio.list_audio_backends():
            torchaudio.set_audio_backend('soundfile')  # Good for WAV/FLAC
                
                # DON'T shuffle here - let DataLoader handle it
        # random.shuffle(self.valid_audio_files)  ← Remove this line

    def __len__(self):
        return len(self.valid_audio_files)
    
    def __getitem__(self, idx):  # ← Changed from __iter__
        # Get the specific audio file at this index
        audio_file = self.valid_audio_files[idx]
        
        try:
            # Get corresponding image file
            image_file = self.audio_to_image_map[audio_file]
            
            # Construct full paths
            audio_path = os.path.join(self.data_base_path, audio_file)
            image_path = os.path.join(self.data_base_path, image_file)
            
            # Check if files exist
            if not os.path.exists(audio_path):
                # Return a different valid sample or raise exception
                print(f"Warning: Audio file not found: {audio_path}")
                # Option 1: Return next valid sample
                return self.__getitem__((idx + 1) % len(self))
                # Option 2: Raise exception (not recommended)
                # raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                return self.__getitem__((idx + 1) % len(self))
            
            # Load and process audio
            waveform, sr = torchaudio.load(audio_path)
            # waveform shape: [channels, time]
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sampling_rate:
                if self.resampler is None or self.resampler.orig_freq != sr:
                    self.resampler = T.Resample(sr, self.sampling_rate)
                waveform = self.resampler(waveform)
            
            audio_tensor = waveform.squeeze(0)
            
            max_samples = int(self.max_audio_duration * self.sampling_rate)
            
            if audio_tensor.shape[0] > max_samples:
                # Random crop if longer than max duration
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
            
            image_tensor, crop_info = process_image(image_path, self.crop_strategy, self.target_size)
            
            return {
                "audio": audio_tensor,
                "audio_attention_mask": attention_mask,
                "sampling_rate": self.sampling_rate,
                "image": image_tensor,
                "audio_path": audio_path,
                "image_path": image_path,
                "crop_info": crop_info
            }
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            # Return next valid sample
            return self.__getitem__((idx + 1) % len(self))



'''
Loaded 2540518 completed audio files
Loading JSON mapping...
Loaded 6795660 audio-image mappings
Found 2540518 valid audio files with image mappings
'''


if __name__ == "__main__":
    dataset = VAAPairedDataset(debug=True)
    import time
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True, prefetch_factor=6)
    for batch in dataloader:
        pass

        