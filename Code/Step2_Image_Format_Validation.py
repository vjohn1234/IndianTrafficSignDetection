#!/usr/bin/env python3
"""
Image Format Validation and Conversion for YOLO Training
Validates all images in the dataset and converts them to proper JPEG format with correct headers.
"""

import os
import cv2
import numpy as np
from PIL import Image
import imghdr
from pathlib import Path
import shutil


class ImageFormatValidator:
    def resize_to_yolo(self, image_path, target_size=640):
        """
        Resize image to YOLO-friendly size (640x640) with aspect-ratio padding (rect=True).
        Overwrites the image in place.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "OpenCV could not read image"
            h, w = img.shape[:2]
            scale = target_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Create a new square image and paste resized image in center
            pad_w = target_size - new_w
            pad_h = target_size - new_h
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            color = [114, 114, 114]  # YOLO default padding color
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            # Save as JPEG (overwrite)
            cv2.imwrite(image_path, padded, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            return True, "Resized to YOLO format"
        except Exception as e:
            return False, f"Resize failed: {str(e)}"
    def __init__(self, dataset_path):
        """Initialize the image format validator."""
        self.dataset_path = dataset_path
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        self.target_format = 'JPEG'
        self.target_extension = '.jpg'
        
        self.conversion_results = {
            'converted_count': 0,
            'failed_conversions': 0,
            'corrupted_removed': 0,
            'conversion_details': []
        }
        
        self.validation_results = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'corrupted_images': 0,
            'wrong_format': 0,
            'header_issues': 0,
            'size_issues': 0,
            'color_issues': 0,
            'issues_found': [],
            'format_distribution': {},
            'size_distribution': {},
            'recommendations': []
        }
    
    def validate_image_header(self, image_path):
        """Validate image header and format."""
        critical_issues = []  # Issues that make the image invalid for YOLO
        warnings = []         # Issues that are just recommendations
        
        try:
            # Check file extension
            file_ext = Path(image_path).suffix.lower()
            if file_ext not in self.supported_formats:
                critical_issues.append(f"Unsupported file extension: {file_ext}")
                return False, critical_issues, warnings
            
            # Detect actual image format using imghdr
            detected_format = imghdr.what(image_path)
            if detected_format is None:
                critical_issues.append("Could not detect image format - corrupted header")
                return False, critical_issues, warnings
            
            # Check if extension matches actual format (CRITICAL ERROR)
            if file_ext in ['.jpg', '.jpeg'] and detected_format != 'jpeg':
                critical_issues.append(f"Extension {file_ext} but detected format is {detected_format}")
            elif file_ext == '.png' and detected_format != 'png':
                critical_issues.append(f"Extension {file_ext} but detected format is {detected_format}")
            elif file_ext == '.bmp' and detected_format != 'bmp':
                critical_issues.append(f"Extension {file_ext} but detected format is {detected_format}")
            elif file_ext == '.webp' and detected_format != 'webp':
                critical_issues.append(f"Extension {file_ext} but detected format is {detected_format}")
            
            # Try to open with PIL for additional validation
            try:
                with Image.open(image_path) as img:
                    # Verify image can be loaded
                    img.verify()
                    
                # Re-open for actual processing (verify() closes the image)
                with Image.open(image_path) as img:
                    # Check image mode (WARNING, not critical)
                    if img.mode not in ['RGB', 'L', 'RGBA']:
                        warnings.append(f"Unusual color mode: {img.mode} (YOLO prefers RGB)")
                    
                    # Check image size
                    width, height = img.size
                    if width < 32 or height < 32:
                        critical_issues.append(f"Image too small: {width}x{height} (minimum recommended: 32x32)")
                    
                    # Large images are WARNING, not critical error
                    if width > 4096 or height > 4096:
                        warnings.append(f"Image very large: {width}x{height} (may cause memory issues)")
                    
                    # Check aspect ratio (WARNING, not critical)
                    aspect_ratio = width / height
                    if aspect_ratio > 10 or aspect_ratio < 0.1:
                        warnings.append(f"Extreme aspect ratio: {aspect_ratio:.2f} (width/height)")
            
            except Exception as e:
                critical_issues.append(f"PIL validation failed: {str(e)}")
                return False, critical_issues, warnings
            
            # Try to open with OpenCV for additional validation
            try:
                img = cv2.imread(image_path)
                if img is None:
                    critical_issues.append("OpenCV could not read image")
                    return False, critical_issues, warnings
                
                # Check if image has valid dimensions
                if len(img.shape) not in [2, 3]:
                    critical_issues.append(f"Invalid image dimensions: {img.shape}")
                    return False, critical_issues, warnings
                
                # Check for empty image
                if img.size == 0:
                    critical_issues.append("Empty image (0 pixels)")
                    return False, critical_issues, warnings
            
            except Exception as e:
                critical_issues.append(f"OpenCV validation failed: {str(e)}")
                return False, critical_issues, warnings
            
            # Image is valid if no critical issues (warnings are OK)
            is_valid = len(critical_issues) == 0
            return is_valid, critical_issues, warnings
            
        except Exception as e:
            critical_issues.append(f"General validation error: {str(e)}")
            return False, critical_issues, warnings
    
    def get_image_info(self, image_path):
        """Get detailed image information."""
        info = {
            'path': image_path,
            'size_bytes': 0,
            'dimensions': (0, 0),
            'format': 'unknown',
            'mode': 'unknown',
            'file_ext': Path(image_path).suffix.lower()
        }
        
        try:
            # File size
            info['size_bytes'] = os.path.getsize(image_path)
            
            # Image details with PIL
            with Image.open(image_path) as img:
                info['dimensions'] = img.size
                info['format'] = img.format
                info['mode'] = img.mode
            
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    def convert_image_to_jpeg(self, image_path, output_path=None, quality=95, resize_yolo=True):
        """
        Convert an image to JPEG format with proper header and quality.
        Returns tuple: (success, error_message, new_path)
        """
        try:
            if output_path is None:
                base_path = os.path.splitext(image_path)[0]
                output_path = base_path + '.jpg'
            with Image.open(image_path) as img:
                if img.mode in ['RGBA', 'P', 'LA']:
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ['RGBA', 'LA'] else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
                with Image.open(output_path) as verify_img:
                    if verify_img.format != 'JPEG':
                        return False, "Conversion failed - output is not JPEG", None
                if image_path != output_path and os.path.exists(image_path):
                    os.remove(image_path)
            # Resize to YOLO format if requested
            if resize_yolo:
                success, msg = self.resize_to_yolo(output_path)
                if not success:
                    return False, f"JPEG conversion ok, but resize failed: {msg}", output_path
            return True, "Conversion and resize successful", output_path
        except Exception as e:
            return False, f"Conversion failed: {str(e)}", None
    
    def fix_all_image_formats(self, backup_originals=True, quality=95, resize_yolo=True):
        """
        Convert all images in the dataset to proper JPEG format.
        """
        print("\n" + "="*60)
        print("🔧 FIXING ALL IMAGE FORMATS")
        print("="*60)
        
        if backup_originals:
            backup_dir = os.path.join(os.path.dirname(self.dataset_path), 'DataSet_Original_Backup')
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
                print(f"📁 Created backup directory: {backup_dir}")
        
        # Get all image files
        image_files = []
        for ext in self.supported_formats:
            pattern = f"*{ext}"
            image_files.extend(Path(self.dataset_path).glob(pattern))
            # Also check uppercase
            pattern = f"*{ext.upper()}"
            image_files.extend(Path(self.dataset_path).glob(pattern))
        
        print(f"📊 Found {len(image_files)} images to process")
        
        converted_count = 0
        failed_count = 0
        skipped_count = 0
        corrupted_removed = 0
        
        for i, image_path in enumerate(image_files, 1):
            if i % 50 == 0 or i == len(image_files):
                print(f"   Processing {i}/{len(image_files)} images...")
            image_path_str = str(image_path)
            try:
                is_valid, critical_errors, warnings = self.validate_image_header(image_path_str)
                if not is_valid and any('corrupted' in error.lower() or 'could not' in error.lower() for error in critical_errors):
                    print(f"🗑️ Removing corrupted image: {image_path.name}")
                    os.remove(image_path_str)
                    corrupted_removed += 1
                    continue
                detected_format = imghdr.what(image_path_str)
                file_ext = image_path.suffix.lower()
                # If already proper JPEG, still resize to YOLO format
                if detected_format == 'jpeg' and file_ext == '.jpg':
                    if resize_yolo:
                        success, msg = self.resize_to_yolo(image_path_str)
                        if success:
                            print(f"🔄 Resized {image_path.name} to YOLO format")
                        else:
                            print(f"⚠️ Resize failed for {image_path.name}: {msg}")
                    skipped_count += 1
                    continue
                if backup_originals:
                    backup_path = os.path.join(backup_dir, image_path.name)
                    if not os.path.exists(backup_path):
                        shutil.copy2(image_path_str, backup_path)
                success, error_msg, new_path = self.convert_image_to_jpeg(image_path_str, quality=quality, resize_yolo=resize_yolo)
                if success:
                    converted_count += 1
                    self.conversion_results['conversion_details'].append({
                        'original': image_path.name,
                        'new': os.path.basename(new_path) if new_path else None,
                        'original_format': detected_format,
                        'status': 'success'
                    })
                    print(f"✅ Converted & resized {image_path.name} ({detected_format} → JPEG)")
                else:
                    failed_count += 1
                    print(f"❌ Failed to convert/resize {image_path.name}: {error_msg}")
                    self.conversion_results['conversion_details'].append({
                        'original': image_path.name,
                        'new': None,
                        'original_format': detected_format,
                        'status': 'failed',
                        'error': error_msg
                    })
            except Exception as e:
                failed_count += 1
                print(f"❌ Error processing {image_path.name}: {str(e)}")
        
        # Update conversion results
        self.conversion_results['converted_count'] = converted_count
        self.conversion_results['failed_conversions'] = failed_count
        self.conversion_results['corrupted_removed'] = corrupted_removed
        
        # Print summary
        print(f"\n📊 Conversion Summary:")
        print(f"   ✅ Successfully converted: {converted_count} images")
        print(f"   ⏭️ Already proper JPEG: {skipped_count} images")
        print(f"   ❌ Failed conversions: {failed_count} images")
        print(f"   🗑️ Corrupted images removed: {corrupted_removed} images")
        print(f"   📁 Total processed: {len(image_files)} images")
        
        if backup_originals and (converted_count > 0 or corrupted_removed > 0):
            print(f"   💾 Original images backed up to: {backup_dir}")
        
        return converted_count, failed_count, corrupted_removed
    
    def validate_directory(self, directory_path):
        """Validate all images in a directory."""
        if not os.path.exists(directory_path):
            print(f"❌ Directory not found: {directory_path}")
            return []
        
        print(f"🔍 Validating images in: {directory_path}")
        
        # Get all image files
        image_files = []
        for ext in self.supported_formats:
            pattern = f"*{ext}"
            image_files.extend(Path(directory_path).glob(pattern))
            # Also check uppercase
            pattern = f"*{ext.upper()}"
            image_files.extend(Path(directory_path).glob(pattern))
        
        print(f"📊 Found {len(image_files)} image files")
        
        results = []
        for i, image_path in enumerate(image_files, 1):
            if i % 50 == 0 or i == len(image_files):
                print(f"   Processing {i}/{len(image_files)} images...")
            
            self.validation_results['total_images'] += 1
            
            # Validate image (now returns errors and warnings separately)
            is_valid, critical_errors, warnings = self.validate_image_header(str(image_path))
            
            # Get image info
            info = self.get_image_info(str(image_path))
            
            # Update statistics
            if is_valid:
                self.validation_results['valid_images'] += 1
            else:
                self.validation_results['invalid_images'] += 1
                
                # Only count critical errors, not warnings
                for error in critical_errors:
                    if 'corrupted' in error.lower() or 'could not' in error.lower():
                        self.validation_results['corrupted_images'] += 1
                    elif 'format' in error.lower() or 'extension' in error.lower():
                        self.validation_results['wrong_format'] += 1
                    elif 'header' in error.lower():
                        self.validation_results['header_issues'] += 1
                    elif 'size' in error.lower() or 'dimensions' in error.lower():
                        self.validation_results['size_issues'] += 1
                    elif 'mode' in error.lower() or 'color' in error.lower():
                        self.validation_results['color_issues'] += 1
                
                self.validation_results['issues_found'].append({
                    'file': str(image_path),
                    'errors': critical_errors,
                    'warnings': warnings,
                    'info': info
                })
            
            # Store warnings separately for valid images too
            if warnings and is_valid:
                self.validation_results['issues_found'].append({
                    'file': str(image_path),
                    'errors': [],
                    'warnings': warnings,
                    'info': info
                })
            
            # Update format distribution
            fmt = info.get('format', 'unknown')
            self.validation_results['format_distribution'][fmt] = \
                self.validation_results['format_distribution'].get(fmt, 0) + 1
            
            # Update size distribution
            dims = info.get('dimensions', (0, 0))
            size_key = f"{dims[0]}x{dims[1]}"
            self.validation_results['size_distribution'][size_key] = \
                self.validation_results['size_distribution'].get(size_key, 0) + 1
            
            results.append({
                'path': str(image_path),
                'valid': is_valid,
                'errors': critical_errors,
                'warnings': warnings,
                'info': info
            })
        
        return results
    
    def validate_after_conversion(self):
        """
        Run validation again after conversion to verify all images are now proper JPEG.
        """
        print("\n" + "="*60)
        print("🔍 POST-CONVERSION VALIDATION")
        print("="*60)
        
        # Reset validation results
        self.validation_results = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'corrupted_images': 0,
            'wrong_format': 0,
            'header_issues': 0,
            'size_issues': 0,
            'color_issues': 0,
            'issues_found': [],
            'format_distribution': {},
            'size_distribution': {},
            'recommendations': []
        }
        
        # Run validation again
        results = self.validate_directory(self.dataset_path)
        self.print_validation_summary()
        
        return self.validation_results
    
    def print_validation_summary(self):
        """Print comprehensive validation summary."""
        print("\n" + "="*60)
        print("📊 IMAGE FORMAT VALIDATION SUMMARY")
        print("="*60)
        
        results = self.validation_results
        
        # Overall statistics
        print(f"\n📈 Overall Statistics:")
        print(f"   Total images scanned: {results['total_images']}")
        print(f"   ✅ Valid images: {results['valid_images']}")
        print(f"   ❌ Invalid images: {results['invalid_images']}")
        
        if results['total_images'] > 0:
            success_rate = (results['valid_images'] / results['total_images']) * 100
            print(f"   📊 Success rate: {success_rate:.1f}%")
        
        # Issue breakdown
        if results['invalid_images'] > 0:
            print(f"\n🔍 Issue Breakdown:")
            if results['corrupted_images'] > 0:
                print(f"   🔥 Corrupted images: {results['corrupted_images']}")
            if results['wrong_format'] > 0:
                print(f"   📝 Wrong format: {results['wrong_format']}")
            if results['header_issues'] > 0:
                print(f"   📄 Header issues: {results['header_issues']}")
            if results['size_issues'] > 0:
                print(f"   📐 Size issues: {results['size_issues']}")
            if results['color_issues'] > 0:
                print(f"   🎨 Color issues: {results['color_issues']}")
        
        # Format distribution
        print(f"\n📋 Format Distribution:")
        for fmt, count in sorted(results['format_distribution'].items()):
            percentage = (count / results['total_images']) * 100
            print(f"   {fmt.upper()}: {count} files ({percentage:.1f}%)")
        
        # Most common sizes
        print(f"\n📏 Most Common Image Sizes:")
        size_items = sorted(results['size_distribution'].items(), 
                          key=lambda x: x[1], reverse=True)[:10]
        for size, count in size_items:
            percentage = (count / results['total_images']) * 100
            print(f"   {size}: {count} files ({percentage:.1f}%)")
        
        # Detailed issues (show first 10 for brevity)
        if results['issues_found']:
            total_with_warnings = len([item for item in results['issues_found'] if item['warnings']])
            total_with_errors = len([item for item in results['issues_found'] if item['errors']])
            
            if total_with_errors > 0:
                print(f"\n❌ Critical Issues Found (showing first 10):")
                error_count = 0
                for i, issue_info in enumerate(results['issues_found']):
                    if issue_info['errors'] and error_count < 10:
                        error_count += 1
                        print(f"\n   {error_count}. File: {Path(issue_info['file']).name}")
                        for error in issue_info['errors']:
                            print(f"      ❌ {error}")
                        
                        info = issue_info['info']
                        if 'dimensions' in info:
                            print(f"      📐 Size: {info['dimensions'][0]}x{info['dimensions'][1]}")
                        if 'format' in info:
                            print(f"      📝 Format: {info['format']}")
                
                if total_with_errors > 10:
                    print(f"\n   ... and {total_with_errors - 10} more critical issues")
            
            if total_with_warnings > 0:
                print(f"\n⚠️ Warnings Found (non-critical, showing first 5):")
                warning_count = 0
                for i, issue_info in enumerate(results['issues_found']):
                    if issue_info['warnings'] and not issue_info['errors'] and warning_count < 5:
                        warning_count += 1
                        print(f"\n   {warning_count}. File: {Path(issue_info['file']).name}")
                        for warning in issue_info['warnings']:
                            print(f"      ⚠️ {warning}")
                        
                        info = issue_info['info']
                        if 'dimensions' in info:
                            print(f"      📐 Size: {info['dimensions'][0]}x{info['dimensions'][1]}")
                
                if total_with_warnings > 5:
                    print(f"\n   ... and {total_with_warnings - 5} more warnings (large images)")
        
        # Recommendations
        self.generate_recommendations()
        if self.validation_results['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in self.validation_results['recommendations']:
                print(f"   • {rec}")
    
    def generate_recommendations(self):
        """Generate recommendations based on validation results."""
        results = self.validation_results
        recommendations = []
        
        if results['invalid_images'] > 0:
            recommendations.append("Fix or remove invalid images before YOLO training")
        
        if results['corrupted_images'] > 0:
            recommendations.append("Re-download or re-capture corrupted images")
        
        # Check format preferences
        jpeg_count = results['format_distribution'].get('JPEG', 0)
        total_images = results['total_images']
        
        if total_images > 0 and jpeg_count / total_images < 0.9:
            recommendations.append("Convert remaining images to JPEG format for consistency")
        
        # Check for warnings about large images
        large_image_warnings = len([item for item in results['issues_found'] 
                                  if any('very large' in w for w in item.get('warnings', []))])
        
        if large_image_warnings > 0:
            recommendations.append(f"Consider resizing {large_image_warnings} large images (>4096px) to reduce memory usage during training")
        
        # Check size consistency
        if len(results['size_distribution']) > 15:
            recommendations.append("Consider resizing images to consistent dimensions for better training")
        
        # Check for very small images
        if results['size_issues'] > 0:
            recommendations.append("Resize very small images (< 32x32) or remove them from training")
        
        # Color mode recommendations
        if results['color_issues'] > 0:
            recommendations.append("Convert images to RGB color mode for optimal YOLO training")
        
        self.validation_results['recommendations'] = recommendations
    
    def save_report(self, output_path):
        """Save detailed validation report to file."""
        try:
            with open(output_path, 'w') as f:
                f.write("IMAGE FORMAT VALIDATION REPORT\n")
                f.write("="*50 + "\n\n")
                
                results = self.validation_results
                
                f.write(f"Total images: {results['total_images']}\n")
                f.write(f"Valid images: {results['valid_images']}\n")
                f.write(f"Invalid images: {results['invalid_images']}\n\n")
                
                f.write("ISSUES FOUND:\n")
                f.write("-" * 20 + "\n")
                for issue_info in results['issues_found']:
                    f.write(f"\nFile: {issue_info['file']}\n")
                    if issue_info.get('errors'):
                        f.write("  CRITICAL ERRORS:\n")
                        for error in issue_info['errors']:
                            f.write(f"    - {error}\n")
                    if issue_info.get('warnings'):
                        f.write("  WARNINGS:\n")
                        for warning in issue_info['warnings']:
                            f.write(f"    - {warning}\n")
                
                f.write(f"\nRECOMMENDATIONS:\n")
                f.write("-" * 20 + "\n")
                for rec in results['recommendations']:
                    f.write(f"- {rec}\n")
            
            print(f"✅ Detailed report saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def save_conversion_report(self, output_path):
        """Save detailed conversion report to file."""
        try:
            with open(output_path, 'w') as f:
                f.write("IMAGE CONVERSION REPORT\n")
                f.write("="*50 + "\n\n")
                
                results = self.conversion_results
                
                f.write(f"Converted images: {results['converted_count']}\n")
                f.write(f"Failed conversions: {results['failed_conversions']}\n")
                f.write(f"Corrupted images removed: {results['corrupted_removed']}\n\n")
                
                f.write("CONVERSION DETAILS:\n")
                f.write("-" * 30 + "\n")
                for detail in results['conversion_details']:
                    f.write(f"\nFile: {detail['original']}\n")
                    f.write(f"Status: {detail['status']}\n")
                    f.write(f"Original format: {detail['original_format']}\n")
                    if detail['new']:
                        f.write(f"New file: {detail['new']}\n")
                    if 'error' in detail:
                        f.write(f"Error: {detail['error']}\n")
            
            print(f"✅ Conversion report saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving conversion report: {e}")


def main():
    """Main validation and conversion function."""
    dataset_path = "/Users/jvarghese/Documents/TrafficSignProject/DataSet"
    
    print("🔍 YOLO Image Format Validation & Conversion")
    print("="*60)
    
    # Initialize validator
    validator = ImageFormatValidator(dataset_path)
    
    # Step 1: Initial validation
    print("\n📋 STEP 1: Initial Validation")
    results = validator.validate_directory(dataset_path)
    validator.print_validation_summary()
    
    # Step 2: Only prompt for conversion/resizing if needed
    all_jpeg = validator.validation_results['valid_images'] == validator.validation_results['total_images']
    all_640 = all(
        info.get('dimensions', (0, 0)) == (640, 640)
        for info in [item['info'] for item in validator.validation_results['issues_found'] if not item.get('errors')]
    )
    if all_jpeg and all_640:
        print("\n✅ All images are already YOLO-ready (JPEG, 640x640). No conversion needed!")
    else:
        print("\n🔧 STEP 2: Convert and Resize Images to YOLO Format (640x640, rect=True)")
        user_choice = input("Proceed with automatic conversion and resizing? (y/n): ").strip().lower()
        if user_choice in ['y', 'yes']:
            converted, failed, removed = validator.fix_all_image_formats(
                backup_originals=True,
                quality=95,
                resize_yolo=True
            )
            # Step 3: Post-conversion validation
            print("\n📋 STEP 3: Post-Conversion Validation")
            validator.validate_after_conversion()
        else:
            print("❌ Conversion and resizing cancelled by user")
    
    # Save detailed report
    report_path = "/Users/jvarghese/Documents/TrafficSignProject/image_validation_report.txt"
    validator.save_report(report_path)
    
    # Save conversion report if conversions were made
    if hasattr(validator, 'conversion_results') and validator.conversion_results['converted_count'] > 0:
        conversion_report_path = "/Users/jvarghese/Documents/TrafficSignProject/image_conversion_report.txt"
        validator.save_conversion_report(conversion_report_path)
    
    return validator.validation_results


if __name__ == "__main__":
    main()