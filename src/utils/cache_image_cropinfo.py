# cache_crop_info_only.py
import torch
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os

def process_single_image(args):
    """Process a single image and return result"""
    image_file, data_base, output_base, crop_strategy, target_size = args
    
    image_path = data_base / image_file
    output_path = output_base / image_file.replace('.jpg', '.pt').replace('.jpeg', '.pt').replace('.png', '.pt')
    
    # Skip if already exists
    if output_path.exists():
        return 'skipped', image_file, None
        
    try:
        # Just open to get dimensions - FAST!
        img = Image.open(image_path)
        width, height = img.size
        img.close()  # Don't need the actual image data
        
        crop_info = {
            "original_width": width,
            "original_height": height,
            "crop_strategy": crop_strategy,
            "target_size": target_size,
            "augmentations_used": False
        }
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save just the crop info
        torch.save({
            'crop_info': crop_info,
            'cached_at': datetime.now().isoformat(),
            'source_path': str(image_path)
        }, output_path)
        
        return 'success', image_file, None
        
    except Exception as e:
        return 'error', image_file, str(e)

def cache_crop_info_only(
    json_dir="/workspace/vaani_jsons",
    data_base_path="/workspace/vaani_data",
    output_dir="/workspace/cached_tensors/images",
    crop_strategy="pad_square",
    target_size=224,
    num_threads=None
):
    """Cache ONLY crop_info metadata without processing images"""
    
    if num_threads is None:
        num_threads = min(32, (os.cpu_count() or 1) + 4)  # Default to reasonable thread count
    
    json_dir = Path(json_dir)
    data_base = Path(data_base_path)
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"Caching crop info only (super fast!)")
    print(f"Output dir: {output_base}")
    print(f"Using {num_threads} threads")
    
    # Collect all unique image paths from JSONs (same as your dataset)
    all_image_files = set()
    
    # Load both train and validation JSONs
    json_patterns = ["train_*.json", "validation_set.json"]
    
    for pattern in json_patterns:
        for json_path in json_dir.glob(pattern):
            print(f"Loading {json_path.name}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                for entry in data:
                    image_file = entry['imageFileName'].lstrip('/')
                    all_image_files.add(image_file)
    
    print(f"Found {len(all_image_files)} unique images")
    
    # Process each image with multithreading
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # Prepare arguments for each worker
    worker_args = [
        (image_file, data_base, output_base, crop_strategy, target_size)
        for image_file in all_image_files
    ]
    
    # Process with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(process_single_image, args): args for args in worker_args}
        
        # Process results as they complete
        with tqdm(total=len(all_image_files), desc="Extracting crop info") as pbar:
            for future in as_completed(future_to_args):
                try:
                    status, image_file, error_msg = future.result()
                    
                    if status == 'success':
                        success_count += 1
                    elif status == 'skipped':
                        skip_count += 1
                    elif status == 'error':
                        error_count += 1
                        print(f"\nError with {image_file}: {error_msg}")
                        
                except Exception as e:
                    error_count += 1
                    print(f"\nUnexpected error: {e}")
                
                pbar.update(1)
    
    print(f"\n✅ Complete!")
    print(f"  Cached: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {success_count + skip_count}")
    
    # Save metadata
    metadata_path = output_base / "crop_info_cache_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "created_at": datetime.now().isoformat(),
            "crop_strategy": crop_strategy,
            "target_size": target_size,
            "total_images": len(all_image_files),
            "cached": success_count,
            "skipped": skip_count,
            "errors": error_count,
            "threads_used": num_threads
        }, f, indent=2)
    
    print(f"\nMetadata saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache only crop info for fast loading")
    parser.add_argument("--json-dir", type=str, default="/workspace/vaani_jsons")
    parser.add_argument("--data-dir", type=str, default="/workspace/vaani_data")
    parser.add_argument("--output-dir", type=str, default="/workspace/cached_tensors/images")
    parser.add_argument("--crop-strategy", type=str, default="pad_square")
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--threads", type=int, default=None, help="Number of threads (default: auto)")
    
    args = parser.parse_args()
    
    cache_crop_info_only(
        json_dir=args.json_dir,
        data_base_path=args.data_dir,
        output_dir=args.output_dir,
        crop_strategy=args.crop_strategy,
        target_size=args.target_size,
        num_threads=args.threads
    )