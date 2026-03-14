#!/usr/bin/env python3.9
"""
Photo Resizer Script
Reduces photo file sizes to under 2MB while maintaining quality
Processes all images in the specified directory
"""

import os
import sys
from PIL import Image, ImageOps
from pathlib import Path
import math

# Configuration
TARGET_DIRECTORY = "/Users/jvarghese/Documents/Car"
MAX_FILE_SIZE_MB = 2.0
MAX_FILE_SIZE_BYTES = int(MAX_FILE_SIZE_MB * 1024 * 1024)
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
OUTPUT_QUALITY = 85  # JPEG quality (1-100)
BACKUP_SUFFIX = "_original"

def get_file_size_mb(file_path):
    """Get file size in MB"""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)

def calculate_resize_factor(current_size_bytes, target_size_bytes):
    """Calculate resize factor to achieve target file size"""
    # Rough estimation: file size is proportional to pixel count
    size_ratio = target_size_bytes / current_size_bytes
    # Take square root since we're dealing with 2D dimensions
    resize_factor = math.sqrt(size_ratio * 0.8)  # 0.8 for safety margin
    return min(resize_factor, 1.0)  # Never upscale

def resize_image_to_target_size(image_path, max_size_bytes, quality=85):
    """Resize image to fit within target file size"""
    try:
        # Open and auto-rotate image
        with Image.open(image_path) as img:
            # Handle EXIF orientation
            img = ImageOps.exif_transpose(img)
            
            # Convert to RGB if necessary (for JPEG saving)
            if img.mode in ('RGBA', 'P', 'LA'):
                # Create white background for transparent images
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            original_size = img.size
            current_size = os.path.getsize(image_path)
            
            print(f"Processing: {os.path.basename(image_path)}")
            print(f"  Original size: {original_size[0]}x{original_size[1]} ({current_size/1024/1024:.2f} MB)")
            
            if current_size <= max_size_bytes:
                print(f"  ✅ Already under {MAX_FILE_SIZE_MB}MB - skipping")
                return True
            
            # Calculate resize factor
            resize_factor = calculate_resize_factor(current_size, max_size_bytes)
            new_width = int(original_size[0] * resize_factor)
            new_height = int(original_size[1] * resize_factor)
            
            print(f"  Resize factor: {resize_factor:.3f}")
            print(f"  New size: {new_width}x{new_height}")
            
            # Create backup of original
            backup_path = image_path.replace('.', f'{BACKUP_SUFFIX}.')
            if not os.path.exists(backup_path):
                os.rename(image_path, backup_path)
                print(f"  📁 Backup created: {os.path.basename(backup_path)}")
            else:
                # Backup already exists, work with original backup
                img = Image.open(backup_path)
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA', 'P', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if img.mode in ('RGBA', 'LA'):
                        background.paste(img, mask=img.split()[-1])
                        img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save with optimization
            save_path = image_path
            resized_img.save(
                save_path, 
                'JPEG', 
                quality=quality, 
                optimize=True,
                progressive=True
            )
            
            # Check final size
            final_size = os.path.getsize(save_path)
            final_size_mb = final_size / (1024 * 1024)
            
            print(f"  ✅ Resized to: {final_size_mb:.2f} MB")
            
            # If still too large, try lower quality
            if final_size > max_size_bytes and quality > 60:
                print(f"  🔄 Still too large, reducing quality...")
                lower_quality = max(60, quality - 15)
                resized_img.save(
                    save_path, 
                    'JPEG', 
                    quality=lower_quality, 
                    optimize=True,
                    progressive=True
                )
                final_size = os.path.getsize(save_path)
                final_size_mb = final_size / (1024 * 1024)
                print(f"  ✅ Final size: {final_size_mb:.2f} MB (quality: {lower_quality})")
            
            return final_size <= max_size_bytes
            
    except Exception as e:
        print(f"  ❌ Error processing {image_path}: {e}")
        return False

def process_directory(directory_path):
    """Process all images in the directory"""
    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return False
    
    print(f"🔍 Scanning directory: {directory_path}")
    
    # Find all image files
    image_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
                # Skip backup files
                if BACKUP_SUFFIX not in file:
                    image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"❌ No image files found in {directory_path}")
        return False
    
    print(f"📊 Found {len(image_files)} image files")
    print(f"🎯 Target size: Under {MAX_FILE_SIZE_MB} MB each")
    print(f"🔧 Output quality: {OUTPUT_QUALITY}")
    print("=" * 60)
    
    # Process each image
    processed = 0
    successful = 0
    
    for image_path in image_files:
        processed += 1
        print(f"\n[{processed}/{len(image_files)}]", end=" ")
        
        if resize_image_to_target_size(image_path, MAX_FILE_SIZE_BYTES, OUTPUT_QUALITY):
            successful += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Processing complete!")
    print(f"📊 Processed: {processed} files")
    print(f"🎉 Successful: {successful} files")
    
    if successful < processed:
        print(f"⚠️  Failed: {processed - successful} files")
    
    return True

def restore_backups(directory_path):
    """Restore original files from backups"""
    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return
    
    backup_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if BACKUP_SUFFIX in file:
                backup_files.append(os.path.join(root, file))
    
    if not backup_files:
        print("❌ No backup files found")
        return
    
    print(f"🔄 Found {len(backup_files)} backup files")
    
    for backup_path in backup_files:
        try:
            original_path = backup_path.replace(f'{BACKUP_SUFFIX}.', '.')
            if os.path.exists(original_path):
                os.remove(original_path)
            os.rename(backup_path, original_path)
            print(f"✅ Restored: {os.path.basename(original_path)}")
        except Exception as e:
            print(f"❌ Failed to restore {backup_path}: {e}")

def main():
    """Main function"""
    print("📸 PHOTO RESIZER")
    print("=" * 60)
    print(f"🎯 Target: Reduce photos to under {MAX_FILE_SIZE_MB} MB")
    print(f"📁 Directory: {TARGET_DIRECTORY}")
    print("=" * 60)
    
    # Check if PIL is available
    try:
        from PIL import Image
    except ImportError:
        print("❌ PIL (Pillow) not found. Install with: pip install Pillow")
        return False
    
    # Command line argument handling
    if len(sys.argv) > 1:
        if sys.argv[1] == "--restore":
            print("🔄 Restoring backups...")
            restore_backups(TARGET_DIRECTORY)
            return True
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python3.9 Resize.py          # Resize photos")
            print("  python3.9 Resize.py --restore # Restore from backups")
            print("  python3.9 Resize.py --help   # Show this help")
            return True
    
    # Process the directory
    success = process_directory(TARGET_DIRECTORY)
    
    if success:
        print(f"\n💡 Tips:")
        print(f"   • Original files backed up with '{BACKUP_SUFFIX}' suffix")
        print(f"   • To restore originals: python3.9 Resize.py --restore")
        print(f"   • Supported formats: {', '.join(SUPPORTED_FORMATS)}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
