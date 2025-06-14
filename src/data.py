#### data.py ####
import argparse
from io import BytesIO
import os
import pickle
from functools import partial
import numpy as np
from datasets import load_dataset, interleave_datasets, Image
from tqdm import tqdm
import torch
from torch.utils.data import IterableDataset
from PIL import Image as PILImage
import random
import torchvision.transforms as transforms


def process_image(image_bytes, crop_strategy="pad_square", target_size=224):
    """
    Args:
        image_bytes: Raw image bytes
        crop_strategy: One of ["stretch", "center_crop", "random_crop", "pad_square"]
        target_size: Target size (224 for ViT)
    
    Returns:
        torch.Tensor: Normalized image tensor [3, 224, 224]
    """
    image = PILImage.open(BytesIO(image_bytes)).convert('RGB')
    
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
    
    def __init__(self, index_path="/home/cis/CisStuff/VeS/image_index.pkl",
                 crop_strategy="pad_square", target_size=224, audio_configs=None,):
        super().__init__()
        if audio_configs is None:
            audio_configs = ["Delhi_NewDelhi", 'WestBengal_Alipurduar', 'WestBengal_CoochBehar', 'WestBengal_DakshinDinajpur', 'WestBengal_Darjeeling', 'WestBengal_Jalpaiguri']#, 'WestBengal_Jhargram', 'WestBengal_Kolkata', 'WestBengal_Malda', 'WestBengal_North24Parganas', 'WestBengal_PaschimMedinipur', 'WestBengal_Purulia', 'Bihar_Araria', 'Bihar_Begusarai',]

        self.crop_strategy = crop_strategy
        self.target_size = target_size
        
        print("Loading image index...")
        with open(index_path, 'rb') as f:
            self.image_index = pickle.load(f)

        print("Loading image dataset (memory-mapped)...")
        self.image_dataset = load_dataset(
            "ARTPARK-IISc/VAANI",
            "images",
            split="train", 
            #cache_dir=cache_dir,
            #num_proc=0,  
            #keep_in_memory=False
        ).cast_column("image", Image(decode=False))
    
        print(f"Loading {len(audio_configs)} audio configs in streaming mode...")
        audio_streams = []
        for config in audio_configs:
            try:
                ds = load_dataset(
                    "ARTPARK-IISc/VAANI",
                    config,
                    split="train",
                    #cache_dir=cache_dir,
                    #streaming=True,
                    #num_proc=12
                )
                audio_streams.append(ds)
                print(f"✓ Loaded {config}")
            except Exception as e:
                print(f"✗ Failed to load {config}: {e}")
        self.interleaved_audio = interleave_datasets(
            audio_streams,
            probabilities=None,  
            seed=69
        )
        #self.interleaved_audio = self.interleaved_audio.with_format("torch")

    def __len__(self):
        return len(self.interleaved_audio)
    
    def __iter__(self):
        for audio_item in self.interleaved_audio:
            img_filename = os.path.basename(audio_item["referenceImage"])
            if img_filename not in self.image_index:
                print(f"Warning: Image {img_filename} not found in index, skipping...")
                continue
            img_idx = self.image_index[img_filename]
            image_item = self.image_dataset[img_idx]
  
            audio_array = audio_item["audio"]["array"]
            sampling_rate = audio_item["audio"]["sampling_rate"]
            max_samples = 5 * sampling_rate  # 5 secs
            
            if len(audio_array) > max_samples:
                start_idx = np.random.randint(0, len(audio_array) - max_samples + 1)
                audio_array = audio_array[start_idx:start_idx + max_samples]

            image_tensor = process_image(
                image_item["image"]['bytes'], 
                self.crop_strategy, 
                self.target_size
            )
            #print(type(audio_array))
            #print(audio_array.shape)
            ##print(type(image_tensor))
            #print(image_tensor.shape)
            yield {
                "audio": audio_array,  # torch.Tensor [1, T]
                "sampling_rate": sampling_rate,  # int
                "image": image_tensor,  # torch.Tensor [3, 224, 224]
            }
