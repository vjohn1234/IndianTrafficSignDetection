#!/usr/bin/env python3
"""
Step 5: YOLO Conversion
Converts XML annotations to YOLO format for training object detection models.
Generates train/val/test splits and class mapping.
"""

import os
import glob
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import sys
import time
from collections import defaultdict

# Try to import yaml, fallback to json if not available
try:
    import yaml
except ImportError:
    yaml = None
    print("Warning: PyYAML not installed. Will use JSON for config file instead.")
    print("To install: pip install pyyaml")


class YOLOConverter:
    def __init__(self, organized_dataset_dir):
        self.organized_dataset_dir = organized_dataset_dir
        self.config_file = os.path.join(organized_dataset_dir, 'batch_labeling_config.json')
        self.progress_file = os.path.join(organized_dataset_dir, 'batch_labeling_progress.json')
        self.augmentation_config_file = os.path.join(organized_dataset_dir, 'augmentation_config.json')
        self.yolo_config_file = os.path.join(organized_dataset_dir, 'yolo_config.json')
        
        # Output directories
        self.yolo_dataset_dir = os.path.join(os.path.dirname(organized_dataset_dir), 'yolo_dataset')
        self.yolo_images_dir = os.path.join(self.yolo_dataset_dir, 'images')
        self.yolo_labels_dir = os.path.join(self.yolo_dataset_dir, 'labels')
        self.yolo_yaml_file = os.path.join(self.yolo_dataset_dir, 'dataset.yaml')
        
        # Data splits directories
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.yolo_images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(self.yolo_labels_dir, split), exist_ok=True)
        
        # Config data
        self.config_data = {}
        self.progress_data = {}
        self.augmentation_data = {}
        self.yolo_data = {
            'class_mapping': {},
            'dataset_stats': {
                'total_classes': 0,
                'total_images': 0,
                'train_images': 0,
                'val_images': 0,
                'test_images': 0,
                'class_distribution': {}
            },
            'data_splits': {
                'train': 0.8,  # Default split ratios
                'val': 0.1,
                'test': 0.1
            }
        }
        
        # Simple class counter for checkpoint tracking
        self.processed_class_count = 0
        
    def load_configuration(self):
        """Load project configuration."""
        print("=== Loading Project Configuration ===")
        
        if not os.path.exists(self.config_file):
            print(f"❌ Configuration file not found: {self.config_file}")
            print("Please run Step2_organize_dataset.py first.")
            return False
        
        with open(self.config_file, 'r') as f:
            self.config_data = json.load(f)
        
        print(f"✅ Loaded configuration for {self.config_data['total_classes']} classes")
        
        return True
    
    def load_progress_data(self):
        """Load labeling progress data."""
        print("\n=== Loading Labeling Progress ===")
        
        if not os.path.exists(self.progress_file):
            print(f"❌ Labeling progress file not found: {self.progress_file}")
            print("Please run Step3_batch_labeling.py first.")
            return False
        
        with open(self.progress_file, 'r') as f:
            self.progress_data = json.load(f)
        
        print(f"✅ Loaded progress data for {self.progress_data['completed_classes']}/{self.config_data['total_classes']} classes")
        print(f"✅ {self.progress_data['completed_images']}/{self.config_data['total_images']} images have been labeled")
        
        return True
    
    def load_augmentation_data(self):
        """Load augmentation configuration data."""
        print("\n=== Loading Augmentation Data ===")
        
        if not os.path.exists(self.augmentation_config_file):
            print(f"❌ Augmentation config file not found: {self.augmentation_config_file}")
            print("Please run Step4_image_augmentation.py first.")
            return False
        
        with open(self.augmentation_config_file, 'r') as f:
            self.augmentation_data = json.load(f)
        
        # Count augmented classes and images
        augmented_classes = len(self.augmentation_data.get('classes', {}))
        total_augmented_images = self.augmentation_data.get('stats', {}).get('total_augmented_images', 0)
        
        print(f"✅ Loaded augmentation data for {augmented_classes} classes")
        print(f"✅ {total_augmented_images} augmented images available")
        
        return True
        
    def load_checkpoint(self):
        """Load checkpoint data if it exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self.checkpoint_data = json.load(f)
                
                completed_count = len(self.checkpoint_data['completed_classes'])
                in_progress = self.checkpoint_data['in_progress_class']
                total_processed = self.checkpoint_data['total_processed_images']
                
                print("\n=== Loading Checkpoint Data ===")
                print(f"✅ Found checkpoint data from {self.checkpoint_data['last_update']}")
                print(f"✅ {completed_count} classes completed")
                if in_progress:
                    print(f"✅ Last class in progress: {in_progress}")
                print(f"✅ {total_processed} images processed so far")
                
                return True
            except Exception as e:
                print(f"⚠️ Error loading checkpoint: {e}")
                return False
        return False
        
    # Simple method to get current progress
    def get_progress_info(self):
        """Get simple progress information."""
        return {
            'processed_class_count': self.processed_class_count,
            'total_classes': len(self.yolo_data.get('class_mapping', {})),
            'total_images': self.yolo_data['dataset_stats']['total_images']
        }
    
    def create_class_mapping(self):
        """Create mapping of class names to YOLO format class IDs."""
        print("\n=== Creating Class Mapping ===")
        
        class_mapping = {}
        class_id = 0
        
        # First mandatory signs (usually with lower sequential IDs)
        for class_info in self.config_data['mandatory_road_signs']['classes']:
            sequential_id = class_info['sequential_id']
            class_name = class_info['class_name']
            class_mapping[sequential_id] = {
                'id': class_id,
                'name': class_name,
                'type': 'mandatory'
            }
            class_id += 1
            
        # Then cautionary signs
        for class_info in self.config_data['cautionary_road_signs']['classes']:
            sequential_id = class_info['sequential_id']
            class_name = class_info['class_name']
            class_mapping[sequential_id] = {
                'id': class_id,
                'name': class_name,
                'type': 'cautionary'
            }
            class_id += 1
        
        self.yolo_data['class_mapping'] = class_mapping
        self.yolo_data['dataset_stats']['total_classes'] = class_id
        
        print(f"✅ Created class mapping for {class_id} classes")
        
        return True
    
    def validate_yolo_annotation(self, yolo_line, image_path, sequential_id):
        """
        Comprehensive validation of YOLO annotation line.
        Returns tuple: (is_valid, error_message)
        """
        try:
            # Check for blank/empty lines
            if not yolo_line or not yolo_line.strip():
                return False, "Empty or blank annotation line"
            
            # Remove any extra whitespace and split
            parts = yolo_line.strip().split()
            
            # Check format: should have exactly 5 parts (class_id x y w h)
            if len(parts) != 5:
                return False, f"Incorrect format: expected 5 values (class_id x y w h), got {len(parts)} values: {parts}"
            
            # Validate class_id
            try:
                class_id = int(parts[0])
                if class_id < 0:
                    return False, f"Negative class ID: {class_id}"
                
                # Check if class_id is within valid range
                max_class_id = len(self.yolo_data['class_mapping']) - 1
                if class_id > max_class_id:
                    return False, f"Class ID {class_id} exceeds maximum valid class ID {max_class_id}"
                    
            except ValueError:
                return False, f"Invalid class ID (not an integer): {parts[0]}"
            
            # Validate coordinates (center_x, center_y, width, height)
            coords = []
            coord_names = ['center_x', 'center_y', 'width', 'height']
            
            for i, (coord_str, coord_name) in enumerate(zip(parts[1:], coord_names)):
                try:
                    coord = float(coord_str)
                    
                    # Check for negative values
                    if coord < 0:
                        return False, f"Negative {coord_name}: {coord}"
                    
                    # Check if coordinates are between 0 and 1 (YOLO format requirement)
                    if coord > 1.0:
                        return False, f"{coord_name} exceeds 1.0: {coord} (YOLO format requires normalized coordinates 0-1)"
                    
                    # Additional checks for width and height (should not be 0)
                    if i >= 2 and coord == 0:  # width or height
                        return False, f"Zero {coord_name}: {coord} (bounding box would be invisible)"
                    
                    coords.append(coord)
                    
                except ValueError:
                    return False, f"Invalid {coord_name} (not a number): {coord_str}"
            
            # Additional geometric validation
            center_x, center_y, width, height = coords
            
            # Check if bounding box extends outside image boundaries
            left = center_x - width/2
            right = center_x + width/2
            top = center_y - height/2
            bottom = center_y + height/2
            
            if left < 0:
                return False, f"Bounding box extends left of image (left={left:.6f})"
            if right > 1:
                return False, f"Bounding box extends right of image (right={right:.6f})"
            if top < 0:
                return False, f"Bounding box extends above image (top={top:.6f})"
            if bottom > 1:
                return False, f"Bounding box extends below image (bottom={bottom:.6f})"
            
            # Check for unreasonably small bounding boxes (likely annotation errors)
            min_size = 0.001  # 0.1% of image
            if width < min_size or height < min_size:
                return False, f"Bounding box too small (width={width:.6f}, height={height:.6f}), minimum size={min_size}"
            
            # Check for unreasonably large bounding boxes (likely annotation errors)
            max_size = 0.95   # 95% of image
            if width > max_size or height > max_size:
                return False, f"Bounding box too large (width={width:.6f}, height={height:.6f}), maximum size={max_size}"
            
            return True, "Valid annotation"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def parse_xml_to_yolo(self, xml_path, image_width, image_height):
        """Convert XML annotation to YOLO format with comprehensive validation."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            yolo_annotations = []
            validation_errors = []
            
            for obj_idx, obj in enumerate(root.findall('object')):
                # Get class name
                class_name_elem = obj.find('name')
                if class_name_elem is None:
                    class_name_elem = obj.find('n')  # Alternative tag
                
                if class_name_elem is None or class_name_elem.text is None:
                    validation_errors.append(f"Object {obj_idx}: Missing class name")
                    continue
                
                class_name = class_name_elem.text.strip()
                if not class_name:
                    validation_errors.append(f"Object {obj_idx}: Empty class name")
                    continue
                
                # Find sequential_id from class name
                sequential_id = None
                for seq_id, info in self.yolo_data['class_mapping'].items():
                    if info['name'] == class_name:
                        sequential_id = seq_id
                        break
                
                if sequential_id is None:
                    validation_errors.append(f"Object {obj_idx}: Unknown class '{class_name}'")
                    continue
                
                # Get bounding box
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    validation_errors.append(f"Object {obj_idx}: Missing bounding box for {class_name}")
                    continue
                
                # Validate bounding box coordinates
                try:
                    xmin_elem = bndbox.find('xmin')
                    ymin_elem = bndbox.find('ymin')
                    xmax_elem = bndbox.find('xmax')
                    ymax_elem = bndbox.find('ymax')
                    
                    if None in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]:
                        validation_errors.append(f"Object {obj_idx}: Missing bounding box coordinates")
                        continue
                    
                    xmin = float(xmin_elem.text)
                    ymin = float(ymin_elem.text)
                    xmax = float(xmax_elem.text)
                    ymax = float(ymax_elem.text)
                    
                    # Validate coordinate values
                    if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
                        validation_errors.append(f"Object {obj_idx}: Negative coordinates (xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax})")
                        continue
                    
                    if xmin >= xmax or ymin >= ymax:
                        validation_errors.append(f"Object {obj_idx}: Invalid bounding box (xmin={xmin} >= xmax={xmax} or ymin={ymin} >= ymax={ymax})")
                        continue
                    
                    if xmax > image_width or ymax > image_height:
                        validation_errors.append(f"Object {obj_idx}: Bounding box extends outside image (image: {image_width}x{image_height}, box: xmax={xmax}, ymax={ymax})")
                        continue
                    
                except (AttributeError, ValueError, TypeError) as e:
                    validation_errors.append(f"Object {obj_idx}: Invalid bounding box values - {e}")
                    continue
                
                # Convert to YOLO format (center_x, center_y, width, height) normalized between 0-1
                center_x = ((xmin + xmax) / 2) / image_width
                center_y = ((ymin + ymax) / 2) / image_height
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height
                
                # Final validation and clipping
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # YOLO class ID
                class_id = self.yolo_data['class_mapping'][sequential_id]['id']
                
                # Create YOLO annotation line
                yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                
                # Validate the generated YOLO annotation
                is_valid, error_msg = self.validate_yolo_annotation(yolo_line, xml_path, sequential_id)
                if not is_valid:
                    validation_errors.append(f"Object {obj_idx}: Generated invalid YOLO annotation - {error_msg}")
                    continue
                
                yolo_annotations.append(yolo_line)
            
            # Report validation errors if any
            if validation_errors:
                print(f"⚠️ Validation warnings for {xml_path}:")
                for error in validation_errors:
                    print(f"   - {error}")
            
            return yolo_annotations
            
        except Exception as e:
            print(f"❌ Error parsing {xml_path}: {str(e)}")
            return []
    
    def validate_saved_yolo_file(self, yolo_file_path, image_path):
        """Validate a saved YOLO annotation file."""
        try:
            with open(yolo_file_path, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                is_valid, error_msg = self.validate_yolo_annotation(line, image_path, "unknown")
                if not is_valid:
                    print(f"❌ Validation failed for {yolo_file_path} line {line_num}: {error_msg}")
                    
        except Exception as e:
            print(f"❌ Error validating saved file {yolo_file_path}: {str(e)}")
    
    def run_final_dataset_validation(self):
        """Run comprehensive validation on the entire generated dataset."""
        print("\n=== Running Final Dataset Validation ===")
        
        validation_summary = {
            'total_images': 0,
            'total_labels': 0,
            'total_annotations': 0,
            'validation_errors': 0,
            'validation_warnings': 0,
            'splits': {
                'train': {'images': 0, 'labels': 0, 'annotations': 0, 'errors': 0},
                'val': {'images': 0, 'labels': 0, 'annotations': 0, 'errors': 0},
                'test': {'images': 0, 'labels': 0, 'annotations': 0, 'errors': 0}
            }
        }
        
        for split in ['train', 'val', 'test']:
            print(f"\n🔍 Validating {split} split...")
            
            images_dir = os.path.join(self.yolo_images_dir, split)
            labels_dir = os.path.join(self.yolo_labels_dir, split)
            
            # Get all image and label files
            image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            
            validation_summary['splits'][split]['images'] = len(image_files)
            validation_summary['splits'][split]['labels'] = len(label_files)
            validation_summary['total_images'] += len(image_files)
            validation_summary['total_labels'] += len(label_files)
            
            # Check for orphaned files
            image_basenames = {f.replace('.jpg', '') for f in image_files}
            label_basenames = {f.replace('.txt', '') for f in label_files}
            
            orphaned_images = image_basenames - label_basenames
            orphaned_labels = label_basenames - image_basenames
            
            if orphaned_images:
                print(f"⚠️ Found {len(orphaned_images)} images without labels in {split}")
                validation_summary['validation_warnings'] += len(orphaned_images)
                
            if orphaned_labels:
                print(f"⚠️ Found {len(orphaned_labels)} labels without images in {split}")
                validation_summary['validation_warnings'] += len(orphaned_labels)
            
            # Validate each label file
            for label_file in label_files:
                label_path = os.path.join(labels_dir, label_file)
                image_path = os.path.join(images_dir, label_file.replace('.txt', '.jpg'))
                
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    file_annotations = 0
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        
                        file_annotations += 1
                        is_valid, error_msg = self.validate_yolo_annotation(line, image_path, "validation")
                        if not is_valid:
                            print(f"❌ {split}/{label_file}:{line_num} - {error_msg}")
                            validation_summary['validation_errors'] += 1
                            validation_summary['splits'][split]['errors'] += 1
                    
                    validation_summary['splits'][split]['annotations'] += file_annotations
                    validation_summary['total_annotations'] += file_annotations
                    
                except Exception as e:
                    print(f"❌ Error reading {label_path}: {str(e)}")
                    validation_summary['validation_errors'] += 1
                    validation_summary['splits'][split]['errors'] += 1
        
        # Print validation summary
        print(f"\n📊 Dataset Validation Summary:")
        print(f"   Total Images: {validation_summary['total_images']}")
        print(f"   Total Label Files: {validation_summary['total_labels']}")
        print(f"   Total Annotations: {validation_summary['total_annotations']}")
        print(f"   Validation Errors: {validation_summary['validation_errors']}")
        print(f"   Validation Warnings: {validation_summary['validation_warnings']}")
        
        for split in ['train', 'val', 'test']:
            split_data = validation_summary['splits'][split]
            print(f"\n   {split.upper()} Split:")
            print(f"     Images: {split_data['images']}")
            print(f"     Labels: {split_data['labels']}")
            print(f"     Annotations: {split_data['annotations']}")
            print(f"     Errors: {split_data['errors']}")
        
        # Check for critical issues
        if validation_summary['validation_errors'] > 0:
            print(f"\n⚠️ Found {validation_summary['validation_errors']} validation errors!")
            print("   Please review and fix these issues before training.")
            return False
        elif validation_summary['validation_warnings'] > 0:
            print(f"\n⚠️ Found {validation_summary['validation_warnings']} validation warnings.")
            print("   Dataset is usable but consider reviewing warnings.")
        else:
            print(f"\n✅ Dataset validation passed! No errors found.")
        
        return True
    
    def convert_annotations(self, resume=False):
        """Convert XML annotations to YOLO format and create train/val/test splits."""
        print("\n=== Converting Annotations to YOLO Format ===")
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # For resuming, we'll identify and fix any mismatches
        if resume:
            print("🔍 Checking for consistency in existing dataset...")
            
            # Check how many classes are already processed by counting files
            try:
                # Check for mismatches in each split
                fixed_files = 0
                for split in ['train', 'val', 'test']:
                    image_files = [f for f in os.listdir(os.path.join(self.yolo_images_dir, split)) if f.endswith('.jpg')]
                    label_files = [f.replace('.txt', '.jpg') for f in os.listdir(os.path.join(self.yolo_labels_dir, split)) if f.endswith('.txt')]
                    
                    # Find images without labels
                    images_without_labels = set(image_files) - set(label_files)
                    
                    # Find labels without images
                    labels_without_images = set(label_files) - set(image_files)
                    
                    if images_without_labels:
                        print(f"⚠️ Found {len(images_without_labels)} images without labels in {split} set")
                        # Delete orphaned images
                        for img in images_without_labels:
                            try:
                                os.remove(os.path.join(self.yolo_images_dir, split, img))
                                fixed_files += 1
                            except Exception as e:
                                print(f"   ❌ Error removing orphaned image {img}: {e}")
                    
                    if labels_without_images:
                        print(f"⚠️ Found {len(labels_without_images)} labels without images in {split} set")
                        # Delete orphaned labels
                        for lbl in labels_without_images:
                            txt_file = lbl.replace('.jpg', '.txt')
                            try:
                                os.remove(os.path.join(self.yolo_labels_dir, split, txt_file))
                                fixed_files += 1
                            except Exception as e:
                                print(f"   ❌ Error removing orphaned label {txt_file}: {e}")
                
                if fixed_files > 0:
                    print(f"✅ Fixed {fixed_files} mismatched files")
                else:
                    print("✅ No mismatches found - dataset is consistent")
                
                # Get processed classes count
                processed_classes_estimate = len(set([f.split('_')[0] for f in os.listdir(os.path.join(self.yolo_labels_dir, 'train')) if f.endswith('.txt')]))
                print(f"✅ Found {processed_classes_estimate} processed classes")
                self.processed_class_count = processed_classes_estimate
            except Exception as e:
                print(f"⚠️ Warning: Could not determine progress: {e}")
                self.processed_class_count = 0
        
        # Get split ratios
        train_ratio = self.yolo_data['data_splits']['train']
        val_ratio = self.yolo_data['data_splits']['val']
        test_ratio = self.yolo_data['data_splits']['test']
        
        # Validate split ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:  # Allow small floating point difference
            print(f"⚠️ Warning: Split ratios do not sum to 1.0: {total_ratio}")
            # Normalize the ratios
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio
            print(f"✅ Normalized ratios: Train {train_ratio:.2f}, Val {val_ratio:.2f}, Test {test_ratio:.2f}")
        
        # Track statistics
        total_images = 0
        train_images = 0
        val_images = 0
        test_images = 0
        class_distribution = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
        
        # Track which original images go to which split
        original_image_assignments = {}
        augmented_image_assignments = {}
        
        # Process each class
        classes_to_process = []
        
        # Add mandatory classes
        for class_info in self.config_data['mandatory_road_signs']['classes']:
            sequential_id = class_info['sequential_id']
            if (sequential_id in self.progress_data.get('mandatory_progress', {}) and 
                self.progress_data['mandatory_progress'][sequential_id].get('status') == 'completed'):
                classes_to_process.append(class_info)
        
        # Add cautionary classes
        for class_info in self.config_data['cautionary_road_signs']['classes']:
            sequential_id = class_info['sequential_id']
            if (sequential_id in self.progress_data.get('cautionary_progress', {}) and 
                self.progress_data['cautionary_progress'][sequential_id].get('status') == 'completed'):
                classes_to_process.append(class_info)
        
        print(f"🔄 Converting annotations for {len(classes_to_process)} classes...")
        
        # Process each class
        for class_info in classes_to_process:
            sequential_id = class_info['sequential_id']
            class_name = class_info['class_name']
            
            # Skip if already processed based on class count
            if resume and int(sequential_id) <= self.processed_class_count:
                print(f"⏩ Skipping {sequential_id}_{class_name} (already processed)")
                continue
                
            print(f"\n🔄 Processing {sequential_id}_{class_name}")
            
            # Determine class type and directory
            class_type = None
            for t in ['mandatory', 'cautionary']:
                for c in self.config_data[f'{t}_road_signs']['classes']:
                    if c['sequential_id'] == sequential_id:
                        class_type = t
                        break
                if class_type:
                    break
            
            if not class_type:
                print(f"⚠️ Warning: Could not determine type for class {sequential_id}_{class_name}")
                continue
            
            # Get paths
            class_dir = f"{sequential_id}_{class_name}"
            category_dir = f"{class_type.capitalize()}_Road_Signs"
            class_path = os.path.join(self.organized_dataset_dir, category_dir, class_dir)
            
            images_dir = os.path.join(class_path, 'images')
            annotations_dir = os.path.join(class_path, 'annotations')
            augmented_images_dir = os.path.join(class_path, 'augmented_images')
            augmented_annotations_dir = os.path.join(class_path, 'augmented_annotations')
            
            # Get original images with annotations
            original_images = []
            for xml_file in glob.glob(os.path.join(annotations_dir, '*.xml')):
                img_file = os.path.basename(xml_file).replace('.xml', '.jpg')
                img_path = os.path.join(images_dir, img_file)
                
                if os.path.exists(img_path):
                    original_images.append((img_path, xml_file))
            
            # Get augmented images with annotations
            augmented_images = []
            for xml_file in glob.glob(os.path.join(augmented_annotations_dir, '*.xml')):
                img_file = os.path.basename(xml_file).replace('.xml', '.jpg')
                img_path = os.path.join(augmented_images_dir, img_file)
                
                if os.path.exists(img_path):
                    augmented_images.append((img_path, xml_file))
            
            # Shuffle and split original images (stratified by class)
            random.shuffle(original_images)
            
            # Determine split indices for original images
            n_original = len(original_images)
            n_train_original = int(n_original * train_ratio)
            n_val_original = int(n_original * val_ratio)
            
            # Split original images
            original_train = original_images[:n_train_original]
            original_val = original_images[n_train_original:n_train_original + n_val_original]
            original_test = original_images[n_train_original + n_val_original:]
            
            # Keep track of which original images go to which split
            for img_path, _ in original_train:
                base_name = os.path.basename(img_path)
                original_image_assignments[f"{sequential_id}_{base_name}"] = 'train'
            
            for img_path, _ in original_val:
                base_name = os.path.basename(img_path)
                original_image_assignments[f"{sequential_id}_{base_name}"] = 'val'
            
            for img_path, _ in original_test:
                base_name = os.path.basename(img_path)
                original_image_assignments[f"{sequential_id}_{base_name}"] = 'test'
            
            # Process augmented images
            # For each augmented image, find the original image it was derived from
            # and put it in the same split as the original
            for img_path, xml_path in augmented_images:
                base_name = os.path.basename(img_path)
                # Extract original image name from augmentation name
                # Typical format: 001_001_aug1_light_rain_snow.jpg
                parts = base_name.split('_aug')
                if len(parts) > 1:
                    original_img_name = f"{parts[0]}.jpg"
                    original_key = f"{sequential_id}_{original_img_name}"
                    
                    if original_key in original_image_assignments:
                        split = original_image_assignments[original_key]
                        augmented_image_assignments[f"{sequential_id}_{base_name}"] = split
                    else:
                        # If original not found, assign randomly following the ratio
                        r = random.random()
                        if r < train_ratio:
                            augmented_image_assignments[f"{sequential_id}_{base_name}"] = 'train'
                        elif r < train_ratio + val_ratio:
                            augmented_image_assignments[f"{sequential_id}_{base_name}"] = 'val'
                        else:
                            augmented_image_assignments[f"{sequential_id}_{base_name}"] = 'test'
            
            # Now process all images (original and augmented) based on assignments
            all_images = original_images + augmented_images
            
            class_stats = {
                'total': len(all_images),
                'original': len(original_images),
                'augmented': len(augmented_images),
                'train': 0,
                'val': 0,
                'test': 0
            }
            
            for img_path, xml_path in all_images:
                base_name = os.path.basename(img_path)
                key = f"{sequential_id}_{base_name}"
                
                # Determine which split this image belongs to
                if key in original_image_assignments:
                    split = original_image_assignments[key]
                elif key in augmented_image_assignments:
                    split = augmented_image_assignments[key]
                else:
                    # Fallback: distribute according to ratio
                    r = random.random()
                    if r < train_ratio:
                        split = 'train'
                    elif r < train_ratio + val_ratio:
                        split = 'val'
                    else:
                        split = 'test'
                
                # Update statistics
                total_images += 1
                class_stats[split] += 1
                class_distribution[sequential_id][split] += 1
                
                if split == 'train':
                    train_images += 1
                elif split == 'val':
                    val_images += 1
                else:  # test
                    test_images += 1
                
                # Create YOLO format output filename
                yolo_image_out = os.path.join(self.yolo_images_dir, split, f"{sequential_id}_{base_name}")
                yolo_label_out = os.path.join(self.yolo_labels_dir, split, f"{sequential_id}_{base_name.replace('.jpg', '.txt')}")
                
                # Copy image
                shutil.copy2(img_path, yolo_image_out)
                
                # Convert and save annotation
                try:
                    # Get image dimensions
                    import cv2
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"⚠️ Warning: Could not read image {img_path}")
                        continue
                    
                    img_height, img_width = img.shape[:2]
                    
                    # Convert XML to YOLO format
                    yolo_annotations = self.parse_xml_to_yolo(xml_path, img_width, img_height)
                    
                    # Save YOLO annotation file
                    with open(yolo_label_out, 'w') as f:
                        for line in yolo_annotations:
                            f.write(f"{line}\n")
                    
                    # Validate the saved file
                    self.validate_saved_yolo_file(yolo_label_out, img_path)
                    
                except Exception as e:
                    print(f"❌ Error processing {img_path}: {str(e)}")
                    continue
            
            print(f"   ✅ Processed {class_stats['total']} images ({class_stats['original']} original + {class_stats['augmented']} augmented)")
            print(f"   ✅ Split: Train {class_stats['train']}, Val {class_stats['val']}, Test {class_stats['test']}")
            
            # Update processed class count
            self.processed_class_count = max(self.processed_class_count, int(sequential_id))
        
        # Update dataset statistics
        self.yolo_data['dataset_stats']['total_images'] = total_images
        self.yolo_data['dataset_stats']['train_images'] = train_images
        self.yolo_data['dataset_stats']['val_images'] = val_images
        self.yolo_data['dataset_stats']['test_images'] = test_images
        self.yolo_data['dataset_stats']['class_distribution'] = {k: dict(v) for k, v in class_distribution.items()}
        
        print(f"\n✅ Conversion complete!")
        print(f"✅ Total images: {total_images}")
        print(f"✅ Train images: {train_images} ({train_images/total_images*100:.1f}%)")
        print(f"✅ Validation images: {val_images} ({val_images/total_images*100:.1f}%)")
        print(f"✅ Test images: {test_images} ({test_images/total_images*100:.1f}%)")
        
        return True
    
    def create_yolo_config(self):
        """Create configuration file for YOLOv5/v8 training."""
        print("\n=== Creating YOLO Configuration Files ===")
        
        # Create a clean dictionary of class names in the correct order
        class_names = {}
        for class_id in sorted(self.yolo_data['class_mapping'].keys(), key=lambda x: self.yolo_data['class_mapping'][x]['id']):
            id_num = self.yolo_data['class_mapping'][class_id]['id']
            name = self.yolo_data['class_mapping'][class_id]['name']
            class_names[id_num] = name
        
        # Create ordered list of class names
        ordered_class_names = [class_names[i] for i in range(len(class_names))]
        
        # Always create a YAML file in the specified format
        try:
            with open(self.yolo_yaml_file, 'w') as f:
                # Write root path to dataset - CRITICAL for YOLO to find images
                f.write(f"# Root path to dataset\n")
                f.write(f"path: {self.yolo_dataset_dir}\n\n")
                
                # Write number of classes
                f.write(f"# Number of classes\n")
                f.write(f"nc: {len(class_names)}\n\n")
                
                # Write class names in the specified format
                f.write("# Class names\n")
                f.write("names: [")
                for i, name in enumerate(ordered_class_names):
                    if i > 0:
                        f.write(", ")
                    if i % 5 == 0 and i > 0:  # Line break every 5 classes for readability
                        f.write("\n        ")
                    f.write(f"'{name}'")
                f.write("]\n\n")
                
                # Write paths to dataset images (relative to 'path')
                f.write("# Paths to dataset images (relative to `path`)\n")
                f.write(f"train: images/train\n")
                f.write(f"val: images/val\n")
                f.write(f"test: images/test\n")
            
            print(f"✅ Created YAML configuration file with {len(class_names)} classes in specified format")
        except Exception as e:
            print(f"❌ Error creating YAML file: {str(e)}")
            return False
        
        return True
    
    def delete_yolo_dataset(self):
        """Delete the YOLO dataset directory if it exists."""
        if os.path.exists(self.yolo_dataset_dir):
            print(f"\n🗑️ Deleting existing YOLO dataset directory: {self.yolo_dataset_dir}")
            try:
                # Use shutil.rmtree to remove the entire directory tree
                shutil.rmtree(self.yolo_dataset_dir)
                print(f"✅ YOLO dataset directory completely deleted")
                return True
            except Exception as e:
                print(f"❌ Error deleting YOLO dataset directory: {str(e)}")
                print(f"❌ Attempting to force delete with more permissions...")
                try:
                    # Try to force delete by changing permissions first
                    import stat
                    def handle_remove_readonly(func, path, exc):
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    
                    shutil.rmtree(self.yolo_dataset_dir, onerror=handle_remove_readonly)
                    print(f"✅ YOLO dataset directory force deleted")
                    return True
                except Exception as e2:
                    print(f"❌ Failed to delete YOLO dataset directory even with force: {str(e2)}")
                    return False
        else:
            print(f"ℹ️ No existing YOLO dataset directory found at {self.yolo_dataset_dir}")
            return True
            
    def cleanup_cache_files(self):
        """Remove any .cache files that might be generated during conversion."""
        print("\n🧹 Cleaning up temporary and cache files...")
        cleaned_files = 0
        
        # Find and remove all .cache files in the YOLO dataset directory
        for root, dirs, files in os.walk(self.yolo_dataset_dir):
            for file in files:
                if file.startswith('.') or file.endswith('.cache'):
                    try:
                        os.remove(os.path.join(root, file))
                        cleaned_files += 1
                    except Exception as e:
                        print(f"⚠️ Could not remove cache file {file}: {str(e)}")
        
        if cleaned_files > 0:
            print(f"✅ Removed {cleaned_files} temporary/cache files")
        else:
            print("✅ No temporary/cache files found")
            
        return True
    
    def run(self, resume=False, clean=False):
        """Run the YOLO conversion process."""
        if not self.load_configuration():
            return False
        
        if not self.load_progress_data():
            return False
        
        if not self.load_augmentation_data():
            return False
        
        # Delete existing YOLO dataset if requested
        if clean and not resume:
            if not self.delete_yolo_dataset():
                print(f"❌ Failed to delete YOLO dataset directory")
                return False
        
        # Always ensure directories exist, even when resuming
        if not os.path.exists(self.yolo_dataset_dir):
            os.makedirs(self.yolo_dataset_dir, exist_ok=True)
            print(f"✅ Created YOLO dataset directory: {self.yolo_dataset_dir}")
        
        # Create split directories if they don't exist
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.yolo_images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(self.yolo_labels_dir, split), exist_ok=True)
        
        if not self.create_class_mapping():
            return False
        
        try:
            if not self.convert_annotations(resume=resume):
                return False
            
            if not self.create_yolo_config():
                return False
            
            # Run comprehensive dataset validation
            if not self.run_final_dataset_validation():
                print("\n❌ Dataset validation failed! Please review errors above.")
                return False
                
            # Clean up any temporary or cache files
            self.cleanup_cache_files()
            
            print(f"\n🎉 YOLO Conversion Complete!")
            print(f"✅ Dataset created at: {self.yolo_dataset_dir}")
            
            print(f"\n📂 YOLO Dataset Structure:")
            print(f"   - images/: Contains JPG image files split into train, val, test subdirectories")
            print(f"   - labels/: Contains YOLO format label files (.txt) with same structure as images/")
            print(f"   - dataset.yaml: Configuration file for YOLOv5/v8 training with dataset paths and class names")
            
            print(f"\n🎯 Next Steps:")
            print(f"1. You can now use this dataset with any modern object detection model")
            print(f"2. The dataset is ready at: {self.yolo_dataset_dir}")
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Process interrupted by user!")
            print(f"✅ Progress: Processed approximately {self.processed_class_count} classes")
            print("✅ You can resume later by running the script again and selecting 'Resume'")
            return False


def main():
    organized_dataset_dir = '/Users/jvarghese/Documents/TrafficSignProject/organized_dataset'
    
    print("=== Step 5: YOLO Conversion ===")
    print("This process will convert labeled traffic sign annotations to YOLO format")
    print("and create train/val/test splits for YOLOv5/v8 training.\n")
    
    # Initialize converter
    converter = YOLOConverter(organized_dataset_dir)
    
    # Check if YOLO dataset already exists
    yolo_exists = os.path.exists(converter.yolo_dataset_dir) and os.path.exists(os.path.join(converter.yolo_images_dir, 'train'))
    resume = False
    clean = False
    
    if yolo_exists:
        # Load minimal configuration to estimate progress
        if converter.load_configuration():
            try:
                # Count images by split
                train_images = [f for f in os.listdir(os.path.join(converter.yolo_images_dir, 'train')) if f.endswith('.jpg')]
                val_images = [f for f in os.listdir(os.path.join(converter.yolo_images_dir, 'val')) if f.endswith('.jpg')]
                test_images = [f for f in os.listdir(os.path.join(converter.yolo_images_dir, 'test')) if f.endswith('.jpg')]
                
                # Count labels by split
                train_labels = [f for f in os.listdir(os.path.join(converter.yolo_labels_dir, 'train')) if f.endswith('.txt')]
                val_labels = [f for f in os.listdir(os.path.join(converter.yolo_labels_dir, 'val')) if f.endswith('.txt')]
                test_labels = [f for f in os.listdir(os.path.join(converter.yolo_labels_dir, 'test')) if f.endswith('.txt')]
                
                # Get class statistics
                class_ids = set()
                # Check for classes.names in config directory first, then in root directory as fallback
                class_names_paths = [
                    os.path.join(converter.yolo_dataset_dir, 'config', 'classes.names'),
                    os.path.join(converter.yolo_dataset_dir, 'classes.names')
                ]
                
                for class_names_path in class_names_paths:
                    if os.path.exists(class_names_path):
                        with open(class_names_path, 'r') as f:
                            class_names = [line.strip() for line in f.readlines()]
                            # Use class names from file if found
                            class_ids = set(range(len(class_names)))
                            break
                
                # If no classes.names found, fall back to image filenames
                if not class_ids:
                    class_ids = set([f.split('_')[0] for f in train_images])
                
                num_classes = len(converter.config_data['mandatory_road_signs']['classes']) + len(converter.config_data['cautionary_road_signs']['classes'])
                
                # Get total counts
                total_images = len(train_images) + len(val_images) + len(test_images)
                total_labels = len(train_labels) + len(val_labels) + len(test_labels)
                
                # Check if we have config files
                yaml_exists = os.path.exists(converter.yolo_yaml_file)
                json_exists = os.path.exists(converter.yolo_yaml_file.replace('.yaml', '.json'))
                
                # Display detailed stats
                print(f"\n📊 Found existing YOLO dataset with progress:")
                print(f"   - Processed classes: {len(class_ids)}/{num_classes} ({len(class_ids)/num_classes*100:.1f}%)")
                print(f"   - Images: {total_images} total ({len(train_images)} train, {len(val_images)} validation, {len(test_images)} test)")
                print(f"   - Labels: {total_labels} total ({len(train_labels)} train, {len(val_labels)} validation, {len(test_labels)} test)")
                
                # If there's a mismatch between images and labels, warn the user
                if total_images != total_labels:
                    print(f"⚠️ Warning: Mismatch between images ({total_images}) and labels ({total_labels})!")
                    print(f"   This will be corrected automatically if you choose to resume.")
                
                # If almost complete, show special message
                if len(class_ids) >= num_classes * 0.9:  # 90% or more
                    print(f"🏁 Almost complete! Only {num_classes - len(class_ids)} classes remaining")
                
                # If folder appears empty or incomplete, suggest starting from scratch
                if total_images <= 5 or not (yaml_exists or json_exists):
                    print(f"\n⚠️ Dataset appears to be empty or incomplete - you may want to start from scratch.")
                
            except Exception as e:
                print(f"⚠️ Could not analyze progress: {e}")
        
        # Ask what to do
        print("\nWhat would you like to do?")
        print("1. Resume from where you left off")
        print("2. Start from scratch (will delete existing YOLO dataset)")
        print("3. Cancel conversion")
        
        while True:
            choice = input("Enter your choice (1-3): ").strip()
            if choice == '1':
                resume = True
                clean = False
                print("\n🔄 Resuming conversion from previous progress")
                break
            elif choice == '2':
                resume = False
                clean = True
                print("\n🧹 Starting from scratch (will delete existing YOLO dataset)")
                break
            elif choice == '3':
                print("\n❌ Conversion cancelled")
                sys.exit(0)
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
    else:
        # No existing YOLO dataset
        print("No existing YOLO dataset found.")
        print("\nWhat would you like to do?")
        print("1. Start conversion")
        print("2. Cancel conversion")
        
        while True:
            choice = input("Enter your choice (1-2): ").strip()
            if choice == '1':
                resume = False
                clean = False  # No need to clean as dataset doesn't exist
                print("\n🔄 Starting conversion from scratch")
                break
            elif choice == '2':
                print("\n❌ Conversion cancelled")
                sys.exit(0)
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
    
    # Ask for confirmation
    confirmation = input(f"\n⚠️ Are you sure you want to continue? (y/n): ").strip().lower()
    if confirmation != 'y':
        print("❌ Conversion cancelled")
        sys.exit(0)
    
    # Run the conversion
    success = converter.run(resume=resume, clean=clean)
    
    if success:
        print("\n🎉 Conversion completed successfully!")
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()