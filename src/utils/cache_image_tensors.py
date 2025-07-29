# cache_image_tensors.py
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import numpy as np
from tqdm import tqdm
import cv2
import logging
from datetime import datetime

# Import your process_image function or define it here
from data import process_image  # Or copy the fast cv2 version below

def process_image_fast(image_path, crop_strategy="pad_square", target_size=224):
    """Fast image processing using OpenCV - deterministic version for caching"""
    import cv2
    import numpy as np
    import torch
    
    # Read with OpenCV (much faster than PIL)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image.shape[:2]
    
    if crop_strategy == "pad_square":
        # Faster padding implementation
        max_dim = max(original_width, original_height)
        if max_dim != original_width or max_dim != original_height:
            # Create padded image
            padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
            paste_y = (max_dim - original_height) // 2
            paste_x = (max_dim - original_width) // 2
            padded[paste_y:paste_y+original_height, paste_x:paste_x+original_width] = image
            image = padded
        
        # Resize using cv2 (faster than PIL)
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    elif crop_strategy == "center_crop":
        # Center crop to square, then resize
        min_dim = min(original_width, original_height)
        left = (original_width - min_dim) // 2
        top = (original_height - min_dim) // 2
        image = image[top:top+min_dim, left:left+min_dim]
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
    elif crop_strategy == "stretch":
        # Just resize to target size
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor and normalize in one go
    image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std
    
    crop_info = {
        "original_width": int(original_width),
        "original_height": int(original_height),
        "crop_strategy": crop_strategy,
        "target_size": int(target_size),
        "augmentations_used": False  # Always False for cached tensors
    }
    
    return image, crop_info


def process_single_image(args):
    """Process a single image and save tensor + metadata"""
    image_path, output_path, crop_strategy, target_size = args
    
    try:
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already cached
        if output_path.exists():
            return True, f"Skipped (exists): {image_path.name}"
        
        # Process image
        image_tensor, crop_info = process_image_fast(
            str(image_path), 
            crop_strategy, 
            target_size
        )
        
        # Save both tensor and metadata
        torch.save({
            'tensor': image_tensor,
            'crop_info': crop_info,
            'path': str(image_path),  # Store original path for reference
            'cached_at': datetime.now().isoformat()
        }, output_path)
        
        return True, f"Cached: {image_path.name}"
        
    except Exception as e:
        return False, f"Failed {image_path.name}: {str(e)}"


def cache_dataset_images(
    json_dir="/workspace/vaani_jsons",
    data_base_path="/workspace/vaani_data", 
    output_dir="/workspace/cached_tensors/images",
    crop_strategy="pad_square",
    target_size=224,
    num_workers=32,
    skip_existing=True
):
    """Cache all images referenced in the dataset JSONs"""
    
    json_dir = Path(json_dir)
    data_base = Path(data_base_path)
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_base / f"caching_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting image tensor caching")
    logging.info(f"JSON dir: {json_dir}")
    logging.info(f"Data base: {data_base}")
    logging.info(f"Output dir: {output_base}")
    logging.info(f"Crop strategy: {crop_strategy}")
    logging.info(f"Target size: {target_size}")
    
    # Collect all unique image paths from JSONs
    all_image_files = set()
    
    # Load both train and validation JSONs
    json_patterns = ["train_*.json", "validation_set.json"]
    
    for pattern in json_patterns:
        for json_path in json_dir.glob(pattern):
            logging.info(f"Loading {json_path.name}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                for entry in data:
                    image_file = entry['imageFileName'].lstrip('/')
                    all_image_files.add(image_file)
    
    logging.info(f"Found {len(all_image_files)} unique images to cache")
    
    # Prepare processing arguments
    processing_args = []
    skipped_count = 0
    
    for image_file in all_image_files:
        image_path = data_base / image_file
        
        # Output path mirrors input structure
        output_path = output_base / image_file.replace('.jpg', '.pt').replace('.jpeg', '.pt').replace('.png', '.pt')
        
        # Check if image exists
        if not image_path.exists():
            logging.warning(f"Image not found: {image_path}")
            continue
            
        # Skip if already cached (optional)
        if skip_existing and output_path.exists():
            skipped_count += 1
            continue
            
        processing_args.append((image_path, output_path, crop_strategy, target_size))
    
    logging.info(f"Skipping {skipped_count} already cached images")
    logging.info(f"Processing {len(processing_args)} images with {num_workers} workers")
    
    # Process in parallel
    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(process_single_image, args): args for args in processing_args}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(processing_args), desc="Caching images") as pbar:
            for future in as_completed(future_to_args):
                success, message = future.result()
                
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    logging.error(message)
                
                pbar.update(1)
                pbar.set_postfix({
                    'success': success_count, 
                    'errors': error_count,
                    'skipped': skipped_count
                })
    
    # Final summary
    total_cached = success_count + skipped_count
    logging.info(f"\nCaching complete!")
    logging.info(f"Total cached: {total_cached}")
    logging.info(f"Newly cached: {success_count}")
    logging.info(f"Already existed: {skipped_count}")
    logging.info(f"Errors: {error_count}")
    logging.info(f"Log saved to: {log_file}")
    
    # Create a metadata file with caching info
    metadata_path = output_base / "cache_metadata.json"
    metadata = {
        "created_at": datetime.now().isoformat(),
        "crop_strategy": crop_strategy,
        "target_size": target_size,
        "total_images": len(all_image_files),
        "cached_images": total_cached,
        "errors": error_count,
        "data_base_path": str(data_base),
        "json_dir": str(json_dir)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Caching complete! Metadata saved to {metadata_path}")


def verify_cache(cache_dir, sample_size=10):
    """Verify cached tensors are loadable and have correct format"""
    cache_dir = Path(cache_dir)
    pt_files = list(cache_dir.rglob("*.pt"))
    
    print(f"\nVerifying cache with {len(pt_files)} total files...")
    
    # Random sample
    import random
    sample_files = random.sample(pt_files, min(sample_size, len(pt_files)))
    
    for pt_file in sample_files:
        try:
            data = torch.load(pt_file, map_location='cpu')
            
            # Check structure
            assert 'tensor' in data, "Missing 'tensor' key"
            assert 'crop_info' in data, "Missing 'crop_info' key"
            
            # Check tensor shape
            tensor = data['tensor']
            assert tensor.shape == (3, 224, 224), f"Wrong shape: {tensor.shape}"
            
            # Check crop_info
            crop_info = data['crop_info']
            required_keys = ['original_width', 'original_height', 'crop_strategy', 'target_size']
            for key in required_keys:
                assert key in crop_info, f"Missing crop_info key: {key}"
            
            print(f"✓ {pt_file.name} - OK (original: {crop_info['original_width']}x{crop_info['original_height']})")
            
        except Exception as e:
            print(f"✗ {pt_file.name} - ERROR: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache image tensors for fast data loading")
    parser.add_argument("--json-dir", type=str, default="/workspace/vaani_jsons",
                        help="Directory containing dataset JSON files")
    parser.add_argument("--data-dir", type=str, default="/workspace/vaani_data",
                        help="Base directory containing image files")
    parser.add_argument("--output-dir", type=str, default="/workspace/cached_tensors/images",
                        help="Output directory for cached tensors")
    parser.add_argument("--crop-strategy", type=str, default="pad_square",
                        choices=["pad_square", "center_crop", "stretch"],
                        help="Image cropping strategy")
    parser.add_argument("--target-size", type=int, default=224,
                        help="Target image size")
    parser.add_argument("--num-workers", type=int, default=32,
                        help="Number of parallel workers")
    parser.add_argument("--verify", action="store_true",
                        help="Verify cached tensors after creation")
    parser.add_argument("--force", action="store_true",
                        help="Force re-cache existing files")
    
    args = parser.parse_args()
    
    # Run caching
    cache_dataset_images(
        json_dir=args.json_dir,
        data_base_path=args.data_dir,
        output_dir=args.output_dir,
        crop_strategy=args.crop_strategy,
        target_size=args.target_size,
        num_workers=args.num_workers,
        skip_existing=not args.force
    )
    
    # Optionally verify
    if args.verify:
        verify_cache(args.output_dir)