#!/usr/bin/env python3
"""
Step 4: Image Augmentation
Augments labeled traffic sign images to increase dataset size and variety.
Applies transformations to both images and their corresponding annotations.
"""

import os
import json
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import time
import cv2
import numpy as np
from math import ceil


class ImageAugmenter:
    def __init__(self, organized_dataset_dir):
        self.organized_dataset_dir = organized_dataset_dir
        self.config_file = os.path.join(organized_dataset_dir, 'batch_labeling_config.json')
        self.progress_file = os.path.join(organized_dataset_dir, 'batch_labeling_progress.json')
        self.augmentation_config_file = os.path.join(organized_dataset_dir, 'augmentation_config.json')
        self.config_data = {}
        self.progress_data = {}
        self.augmentation_data = {}
        self.target_images_per_class = 500  # Target total images per class (original + augmented)
    
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
    
    def create_augmentation_config(self):
        """Create or load augmentation configuration."""
        print("\n=== Setting Up Augmentation Configuration ===")
        
        if os.path.exists(self.augmentation_config_file):
            with open(self.augmentation_config_file, 'r') as f:
                self.augmentation_data = json.load(f)
            print(f"✅ Loaded existing augmentation configuration")
        else:
            # Create new augmentation configuration
            print(f"📝 Creating new augmentation configuration")
            
            self.augmentation_data = {
                "creation_date": time.strftime('%Y-%m-%d %H:%M:%S'),
                "target_images_per_class": self.target_images_per_class,
                "transformations": [
                    # Basic transformations
                    "brightness_contrast",
                    "rotation",
                    "flip_horizontal",
                    "gaussian_blur",
                    "add_noise",
                    
                    # Advanced driving condition transformations
                    "rain",
                    "snow",
                    "motion_blur",
                    "glare",
                    "occlusion",
                    "tilt",
                    "night_time",
                    "broken_sign",
                    "shadow"
                ],
                "classes": {},
                "stats": {
                    "total_original_images": self.progress_data['completed_images'],
                    "total_augmented_images": 0,
                    "total_images_after_augmentation": 0
                }
            }
            
            # Display augmentation settings
            print(f"📊 Augmentation Settings:")
            print(f"   - Target images per class: {self.target_images_per_class} (original + augmented)")
            print(f"   - Basic transformations: brightness/contrast, rotation, flip, blur, noise")
            print(f"   - Driving conditions: rain, snow, motion blur, glare, occlusion, tilt")
            print(f"   - Environmental factors: night time, broken sign, shadows")
            
            # Ask for confirmation
            if input(f"\n⚠️  Proceed with these augmentation settings? [y/n]: ").strip().lower() != 'y':
                print(f"🛑 Augmentation cancelled")
                return False
            
            # Save configuration
            self.save_augmentation_config()
        
        return True
    
    def save_augmentation_config(self):
        """Save augmentation configuration to file."""
        with open(self.augmentation_config_file, 'w') as f:
            json.dump(self.augmentation_data, f, indent=2)
        print(f"✅ Saved augmentation configuration")
    
    def get_classes_to_augment(self):
        """Get list of classes that have been labeled and need augmentation."""
        classes_to_augment = []
        
        # Check mandatory classes
        for class_info in self.config_data['mandatory_road_signs']['classes']:
            sequential_id = class_info['sequential_id']
            if (sequential_id in self.progress_data.get('mandatory_progress', {}) and 
                self.progress_data['mandatory_progress'][sequential_id].get('status') == 'completed'):
                classes_to_augment.append(class_info)
        
        # Check cautionary classes
        for class_info in self.config_data['cautionary_road_signs']['classes']:
            sequential_id = class_info['sequential_id']
            if (sequential_id in self.progress_data.get('cautionary_progress', {}) and 
                self.progress_data['cautionary_progress'][sequential_id].get('status') == 'completed'):
                classes_to_augment.append(class_info)
        
        return classes_to_augment
    
    def display_augmentation_menu(self, classes_to_augment):
        """Display menu of classes to augment."""
        print("\n" + "="*60)
        print("           🔄  IMAGE AUGMENTATION SYSTEM  🔄")
        print("="*60)
        
        print(f"\n📋 Classes Available for Augmentation ({len(classes_to_augment)}):")
        
        for i, class_info in enumerate(classes_to_augment, 1):
            sequential_id = class_info['sequential_id']
            class_name = class_info['class_name']
            image_count = class_info['image_count']
            
            # Check if class has already been augmented
            augmented = sequential_id in self.augmentation_data.get("classes", {})
            status = "✅ Augmented" if augmented else "⏳ Not Augmented"
            
            print(f"   {i}. {sequential_id}_{class_name} ({image_count} images) - {status}")
        
        print(f"\n📊 Augmentation Options:")
        print(f"   1. Augment All Classes")
        print(f"   2. Augment Specific Class")
        print(f"   3. View Augmentation Stats")
        print(f"   4. Exit")
        
        return input(f"\n🎯 Choose an option (1-4): ").strip()
    
    def augment_class(self, class_info):
        """Apply augmentation to all labeled images in a class with severity levels."""
        sequential_id = class_info['sequential_id']
        class_name = class_info['class_name']
        images_dir = class_info['images_dir']
        annotations_dir = class_info['annotations_dir']
        
        # Get augmented directories
        augmented_images_dir = os.path.join(os.path.dirname(images_dir), 'augmented_images')
        augmented_annotations_dir = os.path.join(os.path.dirname(annotations_dir), 'augmented_annotations')
        
        # Clean up any existing augmented files
        print(f"\n🧹 Cleaning up existing augmented files for {sequential_id}_{class_name}...")
        if os.path.exists(augmented_images_dir):
            for file in os.listdir(augmented_images_dir):
                os.remove(os.path.join(augmented_images_dir, file))
            print(f"   ✅ Cleaned {augmented_images_dir}")
        
        if os.path.exists(augmented_annotations_dir):
            for file in os.listdir(augmented_annotations_dir):
                os.remove(os.path.join(augmented_annotations_dir, file))
            print(f"   ✅ Cleaned {augmented_annotations_dir}")
        
        # Create directories if they don't exist
        os.makedirs(augmented_images_dir, exist_ok=True)
        os.makedirs(augmented_annotations_dir, exist_ok=True)
        
        # Get list of images with annotations
        annotations = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
        image_files = []
        
        for annotation in annotations:
            img_file = annotation.replace('.xml', '.jpg')
            img_path = os.path.join(images_dir, img_file)
            
            if os.path.exists(img_path):
                image_files.append(img_file)
        
        # Calculate how many augmentations we need per image to reach target
        original_count = len(image_files)
        
        if original_count == 0:
            print(f"❌ No annotated images found for {sequential_id}_{class_name}. Skipping.")
            return 0
        
        # Calculate augmentations needed to reach target
        augmentations_needed = max(0, self.target_images_per_class - original_count)
        
        # Calculate augmentations per image (rounded up to ensure we meet target)
        augmentations_per_image = int(np.ceil(augmentations_needed / original_count))
        
        print(f"\n🔄 Augmenting {sequential_id}_{class_name}")
        print(f"   Original images: {original_count}")
        print(f"   Target total: {self.target_images_per_class}")
        print(f"   Will create ~{augmentations_per_image} augmentations per image")
        
        augmented_count = 0
        severity_stats = {
            'light': 0,
            'medium': 0,
            'heavy': 0
        }
        condition_stats = {}
        
        # Validation statistics
        validation_stats = {
            'total_attempted': 0,
            'validation_passed': 0,
            'validation_failed': 0,
            'validation_warnings': 0,
            'yolo_boundary_rejections': 0  # Track rejections due to boundary issues
        }
        
        # Define realistic driving condition types for more variety
        # Only include conditions that have corresponding methods
        driving_condition_types = [
            "normal",
            "rain",
            "snow", 
            "motion_blur",
            "glare",
            "occlusion",
            "tilt",
            "night_time",
            "broken_sign",
            "shadow"
        ]
        
        # Add fog if the method exists
        if hasattr(self, '_apply_fog_effect'):
            driving_condition_types.append("fog")
        
        # Process each image
        for img_file in image_files:
            xml_file = img_file.replace('.jpg', '.xml')
            img_path = os.path.join(images_dir, img_file)
            xml_path = os.path.join(annotations_dir, xml_file)
            
            print(f"   Processing {img_file}...", end='', flush=True)
            
            try:
                # Load image and annotation with enhanced error handling
                try:
                    image = cv2.imread(img_path)
                    if image is None or image.size == 0:
                        print(f" ❌ Failed to load image or empty image")
                        continue
                        
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                except Exception as e:
                    print(f" ❌ Error loading image or XML: {e}")
                    continue
                
                # Get bounding box
                bbox = self._get_bounding_box_from_xml(root)
                if bbox is None:
                    print(f" ❌ Failed to extract valid bounding box from XML")
                    continue
                    
                # Make sure bbox dimensions are valid
                if bbox['xmin'] >= bbox['xmax'] or bbox['ymin'] >= bbox['ymax']:
                    print(f" ❌ Invalid bounding box dimensions: {bbox}")
                    continue
                
                # Apply realistic driving conditions augmentations
                aug_count = 0
                for i in range(augmentations_per_image):
                    # Choose a global severity for this augmentation
                    severity = random.choice(['light', 'medium', 'heavy'])
                    severity_stats[severity] += 1
                    
                    # Randomly choose 2-3 driving condition types for each augmentation
                    chosen_conditions = random.sample(
                        driving_condition_types, 
                        k=random.randint(2, 3)
                    )
                    
                    # Update condition stats
                    for condition in chosen_conditions:
                        if condition not in condition_stats:
                            condition_stats[condition] = 0
                        condition_stats[condition] += 1
                    
                    # Apply augmentation with selected severity
                    aug_img, aug_bbox = self._apply_realistic_augmentations(
                        image.copy(), 
                        bbox.copy(), 
                        chosen_conditions
                    )
                    
                    # Create condition tag for file naming
                    condition_tag = "_".join(chosen_conditions)
                    
                    # EARLY VALIDATION: Check if augmentation created boundary issues before saving
                    is_yolo_compatible, yolo_errors, _ = self.validate_for_yolo_conversion(aug_bbox, aug_img.shape, f"aug{i+1}_{severity}_{condition_tag}")
                    
                    if not is_yolo_compatible:
                        print(f"   ⚠️ Skipping augmentation {i+1} - would cause Step 5 YOLO errors:")
                        for error in yolo_errors[:2]:  # Show first 2 errors only
                            print(f"      - {error}")
                        validation_stats['yolo_boundary_rejections'] += 1
                        continue  # Skip this augmentation and try the next one
                    
                    # Create augmented file names with condition info and severity level
                    aug_img_file = f"{img_file.split('.')[0]}_aug{i+1}_{severity}_{condition_tag}.jpg"
                    aug_xml_file = f"{xml_file.split('.')[0]}_aug{i+1}_{severity}_{condition_tag}.xml"
                    
                    # Save augmented image
                    aug_img_path = os.path.join(augmented_images_dir, aug_img_file)
                    cv2.imwrite(aug_img_path, aug_img)
                    
                    # Save augmented annotation with validation
                    aug_xml_path = os.path.join(augmented_annotations_dir, aug_xml_file)
                    validation_stats['total_attempted'] += 1
                    
                    # Save augmented annotation first
                    annotation_saved = self._save_augmented_annotation(xml_path, aug_xml_path, aug_bbox, aug_img_file, aug_img.shape)
                    
                    if annotation_saved:
                        # Now run comprehensive XML validation on the saved file
                        is_valid, errors, warnings = self.validate_xml_annotation(aug_xml_path, aug_img_path)
                        
                        if warnings:
                            validation_stats['validation_warnings'] += len(warnings)
                        
                        if is_valid:
                            validation_stats['validation_passed'] += 1
                            aug_count += 1
                            augmented_count += 1
                        else:
                            validation_stats['validation_failed'] += 1
                            print(f"   ❌ XML validation failed for {aug_img_file}:")
                            for error in errors:
                                print(f"      - {error}")
                            # Remove both the annotation and corresponding image
                            try:
                                if os.path.exists(aug_xml_path):
                                    os.remove(aug_xml_path)
                                if os.path.exists(aug_img_path):
                                    os.remove(aug_img_path)
                            except Exception as e:
                                print(f"   ⚠️ Could not remove invalid files {aug_img_file}: {e}")
                            continue
                    else:
                        validation_stats['validation_failed'] += 1
                        # Remove the corresponding image
                        try:
                            if os.path.exists(aug_img_path):
                                os.remove(aug_img_path)
                        except Exception as e:
                            print(f"   ⚠️ Could not remove invalid image {aug_img_file}: {e}")
                        continue
                    
                    # Stop if we've reached our target
                    if original_count + augmented_count >= self.target_images_per_class:
                        break
                
                print(f" ✅ Created {aug_count} augmentations")
                
                # If we've reached our target for this class, break
                if original_count + augmented_count >= self.target_images_per_class:
                    print(f"   📊 Reached target of {self.target_images_per_class} images. Stopping.")
                    break
                
            except Exception as e:
                print(f" ❌ Error: {str(e)}")
        
        # Update augmentation data for this class with severity stats
        if "classes" not in self.augmentation_data:
            self.augmentation_data["classes"] = {}
        
        self.augmentation_data["classes"][sequential_id] = {
            "sequential_id": sequential_id,
            "class_name": class_name,
            "original_image_count": original_count,
            "augmented_image_count": augmented_count,
            "total_image_count": original_count + augmented_count,
            "severity_stats": severity_stats,
            "condition_stats": condition_stats,
            "augmentation_date": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Update total stats
        self.augmentation_data["stats"]["total_augmented_images"] += augmented_count
        
        # Update severity stats in global stats
        if "severity_stats" not in self.augmentation_data["stats"]:
            self.augmentation_data["stats"]["severity_stats"] = {
                "light": 0, 
                "medium": 0, 
                "heavy": 0
            }
            
        for severity, count in severity_stats.items():
            self.augmentation_data["stats"]["severity_stats"][severity] += count
            
        # Update condition stats in global stats
        if "condition_stats" not in self.augmentation_data["stats"]:
            self.augmentation_data["stats"]["condition_stats"] = {}
            
        for condition, count in condition_stats.items():
            if condition not in self.augmentation_data["stats"]["condition_stats"]:
                self.augmentation_data["stats"]["condition_stats"][condition] = 0
            self.augmentation_data["stats"]["condition_stats"][condition] += count
        
        # Save updated config
        self.save_augmentation_config()
        
        print(f"\n✅ Augmentation complete for {sequential_id}_{class_name}")
        print(f"✅ Created {augmented_count} augmented images and annotations")
        print(f"✅ Severity distribution: Light: {severity_stats['light']}, Medium: {severity_stats['medium']}, Heavy: {severity_stats['heavy']}")
        print(f"✅ Total dataset size for this class: {original_count + augmented_count} images")
        
        # Enhanced validation statistics report (similar to Step 3)
        print(f"\n📊 Comprehensive Validation Report:")
        print(f"   • Total augmentations attempted: {validation_stats['total_attempted']}")
        print(f"   • Validation passed: {validation_stats['validation_passed']}")
        print(f"   • Validation failed: {validation_stats['validation_failed']}")
        print(f"   • Validation warnings: {validation_stats['validation_warnings']}")
        
        if validation_stats['total_attempted'] > 0:
            success_rate = (validation_stats['validation_passed'] / validation_stats['total_attempted']) * 100
            print(f"   • Success rate: {success_rate:.1f}%")
            
            # Detailed validation status reporting (like Step 3)
            print(f"")
            
            # File Integrity Status
            file_integrity_status = "✅" if validation_stats['validation_failed'] == 0 else "❌"
            print(f"  {file_integrity_status} File Integrity Status:")
            
            xml_generation = '✅ All generated successfully' if validation_stats['validation_failed'] == 0 else f'❌ {validation_stats["validation_failed"]} failed generation'
            print(f"     • XML Generation: {xml_generation}")
            
            xml_parsing = '✅ All parseable' if validation_stats['validation_failed'] == 0 else f'❌ {validation_stats["validation_failed"]} unparseable'
            print(f"     • XML Parsing: {xml_parsing}")
            
            # Coordinate Validation Status
            coord_status = "✅" if validation_stats['validation_failed'] == 0 else "❌"
            print(f"")
            print(f"  {coord_status} Coordinate Validation Status:")
            
            bbox_structure = '✅ All complete' if validation_stats['validation_failed'] == 0 else f'❌ {validation_stats["validation_failed"]} issues found'
            print(f"     • Bounding Box Structure: {bbox_structure}")
            
            coord_values = '✅ All numeric & valid' if validation_stats['validation_failed'] == 0 else '❌ Invalid values detected'
            print(f"     • Coordinate Values: {coord_values}")
            
            image_bounds = '✅ Within boundaries' if validation_stats['validation_failed'] == 0 else '❌ Boundary violations found'
            print(f"     • Image Bounds: {image_bounds}")
            
            size_requirements = '✅ Proper dimensions' if validation_stats['validation_failed'] == 0 else '❌ Size issues detected'
            print(f"     • Size Requirements: {size_requirements}")
            
            # Data Quality Status
            quality_status = "✅" if validation_stats['validation_warnings'] == 0 else "⚠️"
            print(f"")
            print(f"  {quality_status} Data Quality Status:")
            
            augmentation_quality = '✅ High quality augmentations' if validation_stats['validation_failed'] == 0 else f'❌ {validation_stats["validation_failed"]} quality issues'
            print(f"     • Augmentation Quality: {augmentation_quality}")
            
            warning_status = '✅ No warnings' if validation_stats['validation_warnings'] == 0 else f'⚠️ {validation_stats["validation_warnings"]} warnings'
            print(f"     • Validation Warnings: {warning_status}")
            
            # YOLO Compatibility Status
            yolo_rejections = validation_stats['yolo_boundary_rejections']
            yolo_status = '✅ All YOLO compatible' if yolo_rejections == 0 else f'⚠️ {yolo_rejections} rejected for boundary issues'
            print(f"     • Step 5 YOLO Compatibility: {yolo_status}")
            
            # Overall Status
            print(f"")
            if validation_stats['validation_failed'] == 0:
                if validation_stats['validation_warnings'] == 0:
                    print(f"  🟢 Overall Status: VALIDATION PASSED - All augmentations are high quality")
                else:
                    print(f"  🟡 Overall Status: VALIDATION PASSED with {validation_stats['validation_warnings']} warnings - Review recommended")
            else:
                print(f"  🔴 Overall Status: VALIDATION ISSUES - {validation_stats['validation_failed']} augmentations rejected for quality")
            
            if validation_stats['validation_failed'] > 0:
                print(f"   ⚠️ {validation_stats['validation_failed']} augmentations were rejected due to validation failures")
            if validation_stats['validation_warnings'] > 0:
                print(f"   ⚠️ {validation_stats['validation_warnings']} validation warnings were generated")
        
        return augmented_count
    
    def _get_bounding_box_from_xml(self, root):
        """Extract bounding box coordinates from XML annotation."""
        bbox = {}
        
        # Get class name
        for obj in root.findall('object'):
            # Try standard 'name' tag first
            name_elem = obj.find('name')
            
            # If not found, try alternate tags like 'n' (seen in some XMLs)
            if name_elem is None:
                name_elem = obj.find('n')  # Check for 'n' tag as alternative
                
            if name_elem is None or name_elem.text is None:
                print(f"Warning: Missing name in object")
                continue
                
            name = name_elem.text
            bbox['class'] = name
            
            bndbox = obj.find('bndbox')
            if bndbox is None:
                print(f"Warning: Missing bndbox for {name}")
                continue
                
            # Safely get coordinates with error checking
            try:
                bbox['xmin'] = int(float(bndbox.find('xmin').text))
                bbox['ymin'] = int(float(bndbox.find('ymin').text))
                bbox['xmax'] = int(float(bndbox.find('xmax').text))
                bbox['ymax'] = int(float(bndbox.find('ymax').text))
                break  # Successfully got a valid bbox
            except (AttributeError, ValueError, TypeError) as e:
                print(f"Warning: Invalid bounding box values: {e}")
                continue
        
        # Check if we have valid bbox
        required_keys = ['class', 'xmin', 'ymin', 'xmax', 'ymax']
        if not all(k in bbox for k in required_keys):
            print(f"Error: Could not extract valid bounding box")
            return None
            
        return bbox
    
    def _apply_realistic_augmentations(self, image, bbox, condition_types):
        """Apply realistic driving condition augmentations with varying severity levels."""
        # First, validate inputs to prevent errors
        if image is None or bbox is None:
            print("Warning: Skipping augmentation - image or bbox is None")
            return image, bbox
            
        # Ensure image has valid dimensions
        if image.size == 0:
            print("Warning: Skipping augmentation - empty image")
            return image, bbox
            
        # Ensure all bbox coordinates are present
        for key in ['xmin', 'ymin', 'xmax', 'ymax', 'class']:
            if key not in bbox:
                print(f"Warning: Skipping augmentation - missing '{key}' in bbox")
                return image, bbox
                
        height, width = image.shape[:2]
        
        # Choose a severity level for this augmentation
        # We can vary the distribution to make medium more common
        severity_distribution = {
            'light': 0.3,    # 30% chance of light
            'medium': 0.4,   # 40% chance of medium
            'heavy': 0.3     # 30% chance of heavy
        }
        
        # First apply basic transformations if needed
        if "normal" in condition_types:
            # Basic transformations
            if random.random() < 0.7:  # 70% chance of rotation
                image, bbox = self._rotate_image(image, bbox)
            
            if random.random() < 0.5:  # 50% chance of brightness/contrast adjustment
                image, bbox = self._adjust_brightness_contrast(image, bbox)
        
        # Then apply special driving condition effects with selected severity
        for condition in condition_types:
            # Choose a severity for this effect based on distribution
            severity = random.choices(
                list(severity_distribution.keys()), 
                list(severity_distribution.values())
            )[0]
            
            try:
                if condition == "rain":
                    # Check if the method accepts severity parameter
                    if 'severity' in self._apply_rain_effect.__code__.co_varnames:
                        image = self._apply_rain_effect(image, severity)
                    else:
                        image = self._apply_rain_effect(image)
                elif condition == "snow":
                    # Check if the method accepts severity parameter
                    if 'severity' in self._apply_snow_effect.__code__.co_varnames:
                        image = self._apply_snow_effect(image, severity)
                    else:
                        image = self._apply_snow_effect(image)
                elif condition == "fog":
                    # Skip fog if method doesn't exist
                    if hasattr(self, '_apply_fog_effect'):
                        if 'severity' in self._apply_fog_effect.__code__.co_varnames:
                            image = self._apply_fog_effect(image, severity)
                        else:
                            image = self._apply_fog_effect(image)
                elif condition == "motion_blur":
                    # Check if the method accepts severity parameter
                    if 'severity' in self._apply_motion_blur.__code__.co_varnames:
                        image = self._apply_motion_blur(image, severity)
                    else:
                        image = self._apply_motion_blur(image)
                elif condition == "glare":
                    # Check if the method accepts severity parameter
                    if 'severity' in self._apply_glare_effect.__code__.co_varnames:
                        image = self._apply_glare_effect(image, severity)
                    else:
                        image = self._apply_glare_effect(image)
                elif condition == "occlusion":
                    # Be extra careful with occlusion since it directly uses bbox
                    if bbox is not None and all(k in bbox for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                        image = self._apply_occlusion(image, bbox)
                    else:
                        print(f"Warning: Skipping occlusion due to invalid bbox")
                elif condition == "tilt":
                    # The tilt function modifies the bbox, so we need to be careful
                    if bbox is not None and all(k in bbox for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                        image, bbox = self._apply_perspective_tilt(image, bbox)
                    else:
                        print(f"Warning: Skipping tilt due to invalid bbox")
                elif condition == "night_time":
                    # Check if the method accepts severity parameter
                    if 'severity' in self._apply_night_time.__code__.co_varnames:
                        image = self._apply_night_time(image, severity)
                    else:
                        image = self._apply_night_time(image)
            except Exception as e:
                print(f"Warning: Error applying {condition} effect: {e}")
                
        # Handle other special effects separately to avoid indentation issues
        for condition in condition_types:
            try:
                if condition == "broken_sign":
                    # Check if the method accepts severity parameter
                    if 'severity' in self._apply_broken_sign_effect.__code__.co_varnames:
                        image = self._apply_broken_sign_effect(image, bbox, severity)
                    else:
                        image = self._apply_broken_sign_effect(image, bbox)
                elif condition == "shadow":
                    # Check if the method accepts severity parameter
                    if 'severity' in self._apply_shadow.__code__.co_varnames:
                        image = self._apply_shadow(image, severity)
                    else:
                        image = self._apply_shadow(image)
            except Exception as e:
                print(f"Warning: Error applying {condition} effect: {e}")
        
        # PREVENTIVE VALIDATION: Apply comprehensive validation standards BEFORE returning
        # This prevents creating invalid augmentations that would fail later validation
        
        # Ensure bbox is within image boundaries with safety margins to prevent edge cases
        safety_margin = 1.0  # 1 pixel margin to prevent boundary precision issues
        bbox['xmin'] = max(safety_margin, float(bbox['xmin']))
        bbox['ymin'] = max(safety_margin, float(bbox['ymin']))
        bbox['xmax'] = min(float(width) - safety_margin, float(bbox['xmax']))
        bbox['ymax'] = min(float(height) - safety_margin, bbox['ymax'])
        
        # Additional validation: ensure bbox has valid dimensions
        if bbox['xmin'] >= bbox['xmax']:
            bbox['xmax'] = bbox['xmin'] + 1.0
        if bbox['ymin'] >= bbox['ymax']:
            bbox['ymax'] = bbox['ymin'] + 1.0
            
            # Final clamp to ensure we don't exceed image boundaries (with safety margin)
        bbox['xmax'] = min(float(width) - safety_margin, bbox['xmax'])
        bbox['ymax'] = min(float(height) - safety_margin, bbox['ymax'])
        
        # COMPREHENSIVE PREVENTIVE VALIDATION using same standards as Steps 3 & 5
        bbox_width = bbox['xmax'] - bbox['xmin']
        bbox_height = bbox['ymax'] - bbox['ymin']
        
        # Check for zero-size bounding boxes (same as Steps 3 & 5)
        if bbox_width == 0 or bbox_height == 0:
            print(f"Warning: Preventing zero-size bounding box during augmentation")
            # Fix by ensuring minimum size
            if bbox_width == 0:
                bbox['xmax'] = bbox['xmin'] + 1.0
            if bbox_height == 0:
                bbox['ymax'] = bbox['ymin'] + 1.0
            # Recalculate after fix
            bbox_width = bbox['xmax'] - bbox['xmin']
            bbox_height = bbox['ymax'] - bbox['ymin']
        
        # Size validation with image-relative thresholds (IDENTICAL to Steps 3 & 5)
        if width > 0 and height > 0:
            width_ratio = bbox_width / width
            height_ratio = bbox_height / height
            
            # Unreasonably small (< 0.1% of image dimension) - SAME as Steps 3 & 5
            min_ratio = 0.001  # 0.1% - IDENTICAL threshold
            if width_ratio < min_ratio or height_ratio < min_ratio:
                print(f"Warning: Preventing too-small bounding box during augmentation (width={width_ratio:.3%}, height={height_ratio:.3%})")
                # Fix by scaling up to minimum size
                if width_ratio < min_ratio:
                    target_width = width * min_ratio
                    center_x = (bbox['xmin'] + bbox['xmax']) / 2
                    bbox['xmin'] = max(safety_margin, center_x - target_width/2)
                    bbox['xmax'] = min(width - safety_margin, center_x + target_width/2)
                
                if height_ratio < min_ratio:
                    target_height = height * min_ratio
                    center_y = (bbox['ymin'] + bbox['ymax']) / 2
                    bbox['ymin'] = max(safety_margin, center_y - target_height/2)
                    bbox['ymax'] = min(height - safety_margin, center_y + target_height/2)
            
            # Unreasonably large (> 95% of image) - SAME as Steps 3 & 5
            max_ratio = 0.95  # 95% - IDENTICAL threshold
            if width_ratio > max_ratio or height_ratio > max_ratio:
                print(f"Warning: Preventing too-large bounding box during augmentation (width={width_ratio:.1%}, height={height_ratio:.1%})")
                # Fix by scaling down to maximum size
                if width_ratio > max_ratio:
                    target_width = width * max_ratio
                    center_x = (bbox['xmin'] + bbox['xmax']) / 2
                    bbox['xmin'] = max(safety_margin, center_x - target_width/2)
                    bbox['xmax'] = min(width - safety_margin, center_x + target_width/2)
                
                if height_ratio > max_ratio:
                    target_height = height * max_ratio
                    center_y = (bbox['ymin'] + bbox['ymax']) / 2
                    bbox['ymin'] = max(safety_margin, center_y - target_height/2)
                    bbox['ymax'] = min(height - safety_margin, center_y + target_height/2)
        
        # Final coordinate validation (ensure all values are valid)
        for coord_name in ['xmin', 'ymin', 'xmax', 'ymax']:
            coord_value = bbox[coord_name]
            # Check for NaN or infinite values (same as Steps 3 & 5)
            if not (coord_value == coord_value):  # NaN check
                print(f"Warning: Preventing NaN {coord_name} during augmentation")
                bbox[coord_name] = 0.0 if coord_name in ['xmin', 'ymin'] else (width if coord_name == 'xmax' else height)
            
            if coord_value == float('inf') or coord_value == float('-inf'):
                print(f"Warning: Preventing infinite {coord_name} during augmentation")
                bbox[coord_name] = 0.0 if coord_name in ['xmin', 'ymin'] else (width if coord_name == 'xmax' else height)
        
        return image, bbox
    
    def _adjust_brightness_contrast(self, image, bbox):
        """Adjust brightness and contrast of the image."""
        # Random brightness and contrast adjustment factors
        alpha = random.uniform(0.7, 1.3)  # Contrast
        beta = random.randint(-30, 30)    # Brightness
        
        # Apply adjustment
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Bbox remains unchanged
        return adjusted, bbox
    
    def _rotate_image(self, image, bbox):
        """Rotate image by a small angle."""
        height, width = image.shape[:2]
        angle = random.uniform(-15, 15)
        
        # Get rotation matrix
        center = ((bbox['xmin'] + bbox['xmax']) / 2, (bbox['ymin'] + bbox['ymax']) / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation to image
        rotated = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        
        # Apply rotation to bbox
        points = [
            [bbox['xmin'], bbox['ymin']],
            [bbox['xmax'], bbox['ymin']],
            [bbox['xmin'], bbox['ymax']],
            [bbox['xmax'], bbox['ymax']]
        ]
        
        # Rotate all four corners
        rotated_points = []
        for point in points:
            # Apply rotation matrix
            px = M[0, 0] * point[0] + M[0, 1] * point[1] + M[0, 2]
            py = M[1, 0] * point[0] + M[1, 1] * point[1] + M[1, 2]
            rotated_points.append([px, py])
        
        # Get new bbox from rotated points with safety margins to prevent edge cases
        safety_margin = 1.0  # 1 pixel margin to prevent boundary precision issues
        min_x = max(safety_margin, min([p[0] for p in rotated_points]))
        min_y = max(safety_margin, min([p[1] for p in rotated_points]))
        max_x = min(float(width) - safety_margin, max([p[0] for p in rotated_points]))
        max_y = min(float(height) - safety_margin, max([p[1] for p in rotated_points]))
        
        # Ensure valid bbox dimensions
        if min_x >= max_x:
            max_x = min_x + 1.0
        if min_y >= max_y:
            max_y = min_y + 1.0
            
        new_bbox = bbox.copy()
        new_bbox['xmin'] = int(min_x)
        new_bbox['ymin'] = int(min_y)
        new_bbox['xmax'] = int(min(width, max_x))
        new_bbox['ymax'] = int(min(height, max_y))
        
        return rotated, new_bbox
    
    def _flip_horizontal(self, image, bbox):
        """Flip image horizontally."""
        width = image.shape[1]
        flipped = cv2.flip(image, 1)  # 1 for horizontal flip
        
        # Update bbox coordinates
        new_bbox = bbox.copy()
        new_bbox['xmin'] = width - bbox['xmax']
        new_bbox['xmax'] = width - bbox['xmin']
        
        return flipped, new_bbox
    
    def _apply_gaussian_blur(self, image, bbox):
        """Apply Gaussian blur to the image."""
        # Random kernel size (odd number)
        kernel_size = random.choice([3, 5, 7])
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Bbox remains unchanged
        return blurred, bbox
    
    def _add_noise(self, image, bbox):
        """Add random noise to the image."""
        height, width, channels = image.shape
        noise = np.zeros((height, width, channels), np.uint8)
        
        # Generate noise
        cv2.randn(noise, 0, 25)  # mean=0, stddev=25
        
        # Add noise to image
        noisy = cv2.add(image, noise)
        
        # Bbox remains unchanged
        return noisy, bbox
    
    def _apply_rain_effect(self, image, severity='medium'):
        """Apply rain effect to image with varying severity levels."""
        # Convert image to float32 before any operations
        image_float = image.astype(np.float32)
        height, width = image_float.shape[:2]
        
        # Define parameters based on severity
        if severity == 'light':
            num_drops_range = (300, 600)  # Fewer drops
            length_range = (2, 7)        # Shorter rain streaks
            rain_alpha = 0.4             # Less intense rain
            tint_factor = 0.05           # Less blue tint
        elif severity == 'medium':
            num_drops_range = (600, 1200)
            length_range = (3, 10)
            rain_alpha = 0.6
            tint_factor = 0.1
        else:  # heavy
            num_drops_range = (1000, 2000)  # More drops
            length_range = (5, 15)          # Longer rain streaks
            rain_alpha = 0.8                # More intense rain
            tint_factor = 0.15              # More blue tint
        
        # Rain layer must be float32 for consistent operations
        rain_layer = np.zeros_like(image_float)
        
        # Generate rain drops
        num_drops = random.randint(num_drops_range[0], num_drops_range[1])
        
        for _ in range(num_drops):
            x = random.randint(0, max(1, width-1))
            y = random.randint(0, max(1, height-1))
            length = random.randint(length_range[0], length_range[1])
            angle = random.uniform(-20, 20)
            
            # Calculate end point
            x2 = int(x + length * np.sin(angle * np.pi / 180))
            y2 = int(y + length * np.cos(angle * np.pi / 180))
            
            # Draw line for raindrop
            if 0 <= x2 < width and 0 <= y2 < height:
                cv2.line(rain_layer, (x, y), (x2, y2), (200.0, 200.0, 255.0), 1)
        
        # Blend rain layer with image
        blurred_rain = cv2.GaussianBlur(rain_layer, (5, 5), 0)
        rainy_image = cv2.addWeighted(image_float, 1 - rain_alpha/2, blurred_rain, rain_alpha, 0)
        
        # Add slight blue tint for rainy atmosphere
        blue_tint = np.full_like(rainy_image, (20.0, 20.0, 0.0), dtype=np.float32)
        rainy_image = cv2.addWeighted(rainy_image, 1.0 - tint_factor, blue_tint, tint_factor, 0)
        
        # Convert back to uint8 only at the end
        return np.clip(rainy_image, 0, 255).astype(np.uint8)
    
    def _apply_snow_effect(self, image, severity='medium'):
        """Apply snow effect to image with varying severity levels."""
        # Convert image to float32 before any operations
        image_float = image.astype(np.float32)
        height, width = image_float.shape[:2]
        
        # Define parameters based on severity
        if severity == 'light':
            num_flakes_range = (300, 800)   # Fewer snowflakes
            size_range = (1, 3)            # Smaller flakes
            alpha = 0.3                    # Less snow intensity
            tint_factor = 0.1              # Less white tint
        elif severity == 'medium':
            num_flakes_range = (600, 1500)
            size_range = (1, 4)
            alpha = 0.4
            tint_factor = 0.15
        else:  # heavy
            num_flakes_range = (1200, 2500)  # More snowflakes
            size_range = (1, 5)              # Larger flakes
            alpha = 0.5                      # More intense snow
            tint_factor = 0.2                # More white tint
        
        # Snow layer must be float32 for consistent operations
        snow_layer = np.zeros_like(image_float)
        
        # Generate snowflakes
        num_flakes = random.randint(num_flakes_range[0], num_flakes_range[1])
        
        for _ in range(num_flakes):
            x = random.randint(0, max(1, width-1))
            y = random.randint(0, max(1, height-1))
            size = random.randint(size_range[0], size_range[1])
            
            # Draw snowflake (white dot)
            cv2.circle(snow_layer, (x, y), size, (255.0, 255.0, 255.0), -1)
        
        # Add snow blur - make sure input is float32
        blurred_snow = cv2.GaussianBlur(snow_layer, (5, 5), 0)
        
        # Blend snow layer with image - all inputs are already float32
        # Use severity-based alpha instead of random
        snowy_image = cv2.addWeighted(image_float, 1 - alpha/2, blurred_snow, alpha, 0)
        
        # Add slight white tint for snowy atmosphere
        white_tint = np.full_like(snowy_image, (235.0, 235.0, 235.0), dtype=np.float32)
        snowy_image = cv2.addWeighted(snowy_image, 1.0 - tint_factor, white_tint, tint_factor, 0)
        
        # Convert back to uint8 only at the end
        return np.clip(snowy_image, 0, 255).astype(np.uint8)
    
    def _apply_motion_blur(self, image, severity='medium'):
        """
        Apply motion blur to simulate camera movement with varying severity levels.
        severity: 'light', 'medium', or 'heavy'
        """
        # Define parameters based on severity
        if severity == 'light':
            kernel_sizes = [5, 7]  # Smaller kernel = less blur
            angle_range = (-30, 30)  # Less extreme angles
        elif severity == 'medium':
            kernel_sizes = [7, 9, 11]
            angle_range = (-45, 45)
        else:  # heavy
            kernel_sizes = [13, 15, 17]  # Larger kernel = more motion blur
            angle_range = (-60, 60)  # More extreme angles
        
        # Create kernel for motion blur
        kernel_size = random.choice(kernel_sizes)
        angle = random.uniform(angle_range[0], angle_range[1])
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # Fill kernel with motion line
        for i in range(kernel_size):
            x = int(center + (i - center) * np.cos(np.radians(angle)))
            y = int(center + (i - center) * np.sin(np.radians(angle)))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        # Normalize kernel
        kernel = kernel / kernel.sum()
        
        # Apply motion blur
        blurred_image = cv2.filter2D(image, -1, kernel)
        
        return blurred_image
    
    def _apply_glare_effect(self, image, severity='medium'):
        """
        Apply glare/sun flare effect with varying severity levels.
        severity: 'light', 'medium', or 'heavy'
        """
        # Convert image to float32 before any operations
        image_float = image.astype(np.float32)
        height, width = image_float.shape[:2]
        
        # Define parameters based on severity
        if severity == 'light':
            glare_alpha = 0.2  # Lower intensity glare
            max_radius_factor = 4  # Smaller glare radius
            blur_size = 31  # Less blur
            intensity_factor = 0.8  # Lower brightness
        elif severity == 'medium':
            glare_alpha = 0.3  # Medium intensity
            max_radius_factor = 3  # Medium glare radius
            blur_size = 51  # Medium blur
            intensity_factor = 1.0  # Standard brightness
        else:  # heavy
            glare_alpha = 0.5  # Higher intensity glare
            max_radius_factor = 2  # Larger glare radius (smaller divisor)
            blur_size = 71  # More blur
            intensity_factor = 1.2  # Higher brightness
        
        # Create glare location (usually top corner)
        # Ensure minimum 1 pixel difference between min and max
        w_min = max(0, width//5)
        w_max = max(w_min + 1, width*4//5)
        h_min = max(0, height//5)
        h_max = max(h_min + 1, height*4//5)
        
        glare_x = random.randint(w_min, w_max)
        glare_y = random.randint(h_min, h_max)
        
        # Create glare overlay (as float32 for consistent operations)
        glare_overlay = np.zeros_like(image_float)
        
                # Draw glare circles with decreasing intensity
        max_radius = max(5, min(width, height) // max_radius_factor)  # Ensure minimum radius of 5
        # Ensure step size is appropriate for the radius range
        step_size = max(1, max_radius // 10)
        for radius in range(max_radius, 0, -step_size):
            intensity = 255.0 * (1 - radius/max_radius)**2 * intensity_factor
            cv2.circle(glare_overlay, (glare_x, glare_y), radius, 
                      (intensity, intensity, intensity), -1)        # Ensure blur size is odd
        blur_size = blur_size + 1 if blur_size % 2 == 0 else blur_size
        
        # Blur the glare
        glare_overlay = cv2.GaussianBlur(glare_overlay, (blur_size, blur_size), 0)
        
        # Blend glare with original image (both already float32)
        result = cv2.addWeighted(image_float, 1.0 - glare_alpha, glare_overlay, glare_alpha, 0)
        
        # Convert back to uint8 only at the end
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_occlusion(self, image, bbox):
        """Apply partial occlusion to the traffic sign."""
        height, width = image.shape[:2]
        
        # Define occlusion area (partial coverage of the sign)
        bbox_width = bbox['xmax'] - bbox['xmin']
        bbox_height = bbox['ymax'] - bbox['ymin']
        
        # Choose occlusion type
        occlusion_type = random.choice(['tree_branch', 'pole', 'random_patch'])
        
        if occlusion_type == 'tree_branch':
            # Simulate tree branch occlusion
            branch_width = random.randint(5, 15)
            branch_start_x = random.randint(0, max(1, width-1))
            # Ensure valid ranges for branch positions
            y_min = max(0, bbox['ymin'])
            branch_start_y = random.randint(0, max(1, y_min))
            y_max = min(height-1, bbox['ymax'])
            # Make sure end_y is always > start_y
            branch_end_y = random.randint(min(height-2, y_max+1), max(y_max+2, height-1))
            
            # Draw branch
            cv2.line(image, (branch_start_x, branch_start_y), 
                    (branch_start_x, branch_end_y), (60, 40, 20), branch_width)
            
            # Draw some leaf-like patterns
            for _ in range(4):
                leaf_x = branch_start_x + random.randint(-20, 20)
                # Ensure valid range for leaf_y
                if branch_start_y >= branch_end_y:  # Safety check
                    leaf_y = branch_start_y
                else:
                    leaf_y = random.randint(branch_start_y, branch_end_y)
                leaf_size = random.randint(5, 15)
                cv2.circle(image, (leaf_x, leaf_y), leaf_size, (40, 90, 20), -1)
            
        elif occlusion_type == 'pole':
            # Simulate pole occlusion
            # Ensure bbox_width is at least 1
            safe_bbox_width = max(1, bbox_width)
            pole_x = bbox['xmin'] + random.randint(0, safe_bbox_width)
            pole_width = random.randint(10, 30)
            
            # Draw pole
            cv2.rectangle(image, (pole_x - pole_width//2, 0), 
                         (pole_x + pole_width//2, height), (90, 90, 90), -1)
            
        else:  # random patch
            # Occlude random area of the sign
            # Ensure min < max for all random ranges
            min_patch_width = max(1, bbox_width//5)
            max_patch_width = max(min_patch_width + 1, bbox_width//2)
            patch_width = random.randint(min_patch_width, max_patch_width)
            
            min_patch_height = max(1, bbox_height//5)
            max_patch_height = max(min_patch_height + 1, bbox_height//2)
            patch_height = random.randint(min_patch_height, max_patch_height)
            
            # Ensure x,y ranges are valid
            x_min = bbox['xmin']
            x_max = max(x_min, bbox['xmax'] - patch_width)
            if x_min >= x_max:  # Handle edge case
                x_min = bbox['xmin']
                x_max = x_min + 1
                patch_width = 1
                
            y_min = bbox['ymin']
            y_max = max(y_min, bbox['ymax'] - patch_height)
            if y_min >= y_max:  # Handle edge case
                y_min = bbox['ymin']
                y_max = y_min + 1
                patch_height = 1
                
            patch_x = random.randint(x_min, x_max)
            patch_y = random.randint(y_min, y_max)
            
            # Create random patch
            color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
            cv2.rectangle(image, (patch_x, patch_y), 
                         (patch_x + patch_width, patch_y + patch_height), color, -1)
        
        return image
    
    def _apply_perspective_tilt(self, image, bbox):
        """Apply perspective transformation to simulate tilted viewing angle."""
        # Safety check to ensure image and bbox are valid
        if image is None or image.size == 0 or bbox is None:
            print("Warning: Invalid image or bbox for perspective tilt")
            return image, bbox
            
        # Make sure we have all required bbox coordinates
        if not all(k in bbox for k in ['xmin', 'ymin', 'xmax', 'ymax']):
            print("Warning: Missing bbox coordinates for perspective tilt")
            return image, bbox
            
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            print("Warning: Invalid image dimensions for perspective tilt")
            return image, bbox
        
        # Define source points - corners of the image
        src_pts = np.float32([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
        
        # Define random perspective change
        tilt_intensity = random.uniform(0.05, 0.2)
        
        # Choose tilt direction
        tilt_type = random.choice(['left', 'right', 'up', 'down'])
        
        if tilt_type == 'left':
            # Tilt left side closer
            dst_pts = np.float32([
                [width*tilt_intensity, height*tilt_intensity], 
                [width-1, 0],
                [width*tilt_intensity, height-1-height*tilt_intensity], 
                [width-1, height-1]
            ])
        elif tilt_type == 'right':
            # Tilt right side closer
            dst_pts = np.float32([
                [0, 0], 
                [width-1-width*tilt_intensity, height*tilt_intensity],
                [0, height-1], 
                [width-1-width*tilt_intensity, height-1-height*tilt_intensity]
            ])
        elif tilt_type == 'up':
            # Tilt top closer
            dst_pts = np.float32([
                [width*tilt_intensity, height*tilt_intensity], 
                [width-1-width*tilt_intensity, height*tilt_intensity],
                [0, height-1], 
                [width-1, height-1]
            ])
        else:  # down
            # Tilt bottom closer
            dst_pts = np.float32([
                [0, 0], 
                [width-1, 0],
                [width*tilt_intensity, height-1-height*tilt_intensity], 
                [width-1-width*tilt_intensity, height-1-height*tilt_intensity]
            ])
        
        # Calculate transformation matrix and apply perspective transformation
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_image = cv2.warpPerspective(image, M, (width, height))
        
        # Transform bbox coordinates
        bbox_points = np.array([
            [bbox['xmin'], bbox['ymin'], 1],
            [bbox['xmax'], bbox['ymin'], 1],
            [bbox['xmin'], bbox['ymax'], 1],
            [bbox['xmax'], bbox['ymax'], 1]
        ])
        
        # Apply transformation to bbox points with error handling
        transformed_points = []
        for point in bbox_points:
            try:
                p = np.dot(M, point)
                # Check for division by zero
                if p[2] != 0:
                    p = p / p[2]  # Normalize by homogeneous coordinate
                    transformed_points.append(p[:2])
                else:
                    # Use original point if division by zero would occur
                    transformed_points.append(point[:2])
            except Exception as e:
                print(f"Warning: Error transforming point: {e}")
                # Use original point in case of error
                transformed_points.append(point[:2])
        
        # Get new bbox from transformed points with safety margins to prevent edge cases
        transformed_points = np.array(transformed_points)
        
        # Define safety margin (1 pixel) to prevent bounding boxes from touching edges
        safety_margin = 1.0
        
        new_xmin = max(safety_margin, float(np.min(transformed_points[:, 0])))
        new_ymin = max(safety_margin, float(np.min(transformed_points[:, 1])))
        new_xmax = min(float(width) - safety_margin, float(np.max(transformed_points[:, 0])))
        new_ymax = min(float(height) - safety_margin, float(np.max(transformed_points[:, 1])))
        
        # Apply floating-point precision correction
        tolerance = 1e-5  # Small tolerance for precision errors
        if new_xmin < 0 and new_xmin >= -tolerance:
            new_xmin = 0.0
        if new_ymin < 0 and new_ymin >= -tolerance:
            new_ymin = 0.0
        if new_xmax > width and new_xmax <= width + tolerance:
            new_xmax = float(width)
        if new_ymax > height and new_ymax <= height + tolerance:
            new_ymax = float(height)
        
        # Ensure valid bbox dimensions
        if new_xmin >= new_xmax:
            new_xmax = new_xmin + 1.0
        if new_ymin >= new_ymax:
            new_ymax = new_ymin + 1.0
            
        # Final bounds check with safety margins
        new_xmax = min(float(width) - safety_margin, new_xmax)
        new_ymax = min(float(height) - safety_margin, new_ymax)
        
        # Update bbox
        bbox['xmin'] = int(new_xmin)
        bbox['ymin'] = int(new_ymin)
        bbox['xmax'] = int(new_xmax)
        bbox['ymax'] = int(new_ymax)
        
        return warped_image, bbox
    
    def _apply_night_time(self, image, severity='medium'):
        """
        Apply night time effect (darkening and slight blue tint) with varying severity levels.
        severity: 'light', 'medium', or 'heavy'
        """
        # Convert to float32 at the beginning
        image_float = image.astype(np.float32)
        
        # Define parameters based on severity
        if severity == 'light':
            darkness = random.uniform(0.65, 0.8)  # Less dark
            blue_intensity = 0.05  # Less blue tint
            noise_level = 10  # Less noise
            vignette_strength = 0.3  # Lighter vignette
        elif severity == 'medium':
            darkness = random.uniform(0.5, 0.7)  # Moderate darkness
            blue_intensity = 0.1  # Moderate blue tint
            noise_level = 15  # Moderate noise
            vignette_strength = 0.5  # Standard vignette
        else:  # heavy
            darkness = random.uniform(0.3, 0.5)  # Very dark
            blue_intensity = 0.15  # Strong blue tint
            noise_level = 20  # More noise
            vignette_strength = 0.7  # Strong vignette effect
        
        # Darken image
        darkened = cv2.convertScaleAbs(image_float, alpha=darkness, beta=0).astype(np.float32)
        
        # Add blue tint for night atmosphere (as float32)
        blue_tint = np.full_like(darkened, (20.0, 10.0, 0.0), dtype=np.float32)  # BGR format
        night_image = cv2.addWeighted(darkened, 1.0 - blue_intensity, blue_tint, blue_intensity, 0)
        
        # Add noise to simulate high ISO (as float32)
        noise = np.zeros_like(night_image, dtype=np.float32)
        cv2.randn(noise, 0, noise_level)
        night_image = cv2.add(night_image, noise)
        
        # Add vignette effect
        rows, cols = night_image.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/3)
        kernel_y = cv2.getGaussianKernel(rows, rows/3)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        
        # Apply vignette (already float32)
        for i in range(3):
            night_image[:,:,i] = night_image[:,:,i] * (mask * vignette_strength + (1.0 - vignette_strength))
        
        # Convert back to uint8 only at the end
        return np.clip(night_image, 0, 255).astype(np.uint8)
    
    def _apply_broken_sign_effect(self, image, bbox, severity='medium'):
        """
        Simulate a damaged or broken traffic sign with varying severity levels.
        severity: 'light', 'medium', or 'heavy'
        """
        # Safety check for bbox
        if bbox is None or not all(k in bbox for k in ['xmin', 'ymin', 'xmax', 'ymax']):
            print("Warning: Invalid bbox in broken_sign_effect")
            return image
            
        # Verify that bbox coordinates are valid
        if bbox['xmin'] >= bbox['xmax'] or bbox['ymin'] >= bbox['ymax']:
            print("Warning: Invalid bbox dimensions in broken_sign_effect")
            return image
            
        sign_width = bbox['xmax'] - bbox['xmin']
        sign_height = bbox['ymax'] - bbox['ymin']
        
        # Choose damage type
        damage_type = random.choice(['crack', 'hole', 'bend', 'fade'])
        
        if damage_type == 'crack':
            # Define parameters based on severity
            if severity == 'light':
                crack_count = random.randint(1, 3)  # Fewer cracks
                branch_range = (0, 2)  # Fewer branches
                thickness = 1  # Thinner cracks
            elif severity == 'medium':
                crack_count = random.randint(2, 5)
                branch_range = (1, 3)
                thickness = 2
            else:  # heavy
                crack_count = random.randint(4, 8)  # More cracks
                branch_range = (2, 5)  # More branches
                thickness = 3  # Thicker cracks
            
            # Add random crack lines
            for _ in range(crack_count):
                # Start point inside bbox
                x1 = random.randint(bbox['xmin'], bbox['xmax'])
                y1 = random.randint(bbox['ymin'], bbox['ymax'])
                
                # End point inside bbox
                x2 = random.randint(bbox['xmin'], bbox['xmax'])
                y2 = random.randint(bbox['ymin'], bbox['ymax'])
                
                # Draw crack line
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), thickness=thickness)
                
                # Add some small branch cracks
                # Ensure branch_range has valid values (min <= max)
                min_branches = max(0, branch_range[0])
                max_branches = max(min_branches, branch_range[1])
                branch_count = random.randint(min_branches, max_branches)
                for _ in range(branch_count):
                    mid_x = (x1 + x2) // 2
                    mid_y = (y1 + y2) // 2
                    # Ensure random ranges are valid
                    width_range = max(1, sign_width//4)
                    height_range = max(1, sign_height//4)
                    branch_x = mid_x + random.randint(-width_range, width_range)
                    branch_y = mid_y + random.randint(-height_range, height_range)
                    cv2.line(image, (mid_x, mid_y), (branch_x, branch_y), (0, 0, 0), thickness=max(1, thickness-1))
        
        elif damage_type == 'hole':
            # Define parameters based on severity
            if severity == 'light':
                hole_radius_range = (sign_width//25, sign_width//15)  # Smaller hole
                irregularity = 4  # Less irregularity
            elif severity == 'medium':
                hole_radius_range = (sign_width//20, sign_width//8)
                irregularity = 6
            else:  # heavy
                hole_radius_range = (sign_width//10, sign_width//5)  # Larger hole
                irregularity = 10  # More irregularity
            
            # Add a hole in the sign
            hole_x = random.randint(bbox['xmin'], bbox['xmax'] - sign_width//5)
            hole_y = random.randint(bbox['ymin'], bbox['ymax'] - sign_height//5)
            hole_radius = random.randint(hole_radius_range[0], hole_radius_range[1])
            
            # Draw hole (black circle)
            cv2.circle(image, (hole_x, hole_y), hole_radius, (0, 0, 0), -1)
            
            # Add some irregularity to the hole
            for _ in range(irregularity):
                angle = random.uniform(0, 2*np.pi)
                r = hole_radius * random.uniform(0.8, 1.2)
                x = int(hole_x + r * np.cos(angle))
                y = int(hole_y + r * np.sin(angle))
                small_radius = random.randint(2, 5)
                cv2.circle(image, (x, y), small_radius, (0, 0, 0), -1)
        
        elif damage_type == 'bend':
            # Define parameters based on severity
            if severity == 'light':
                warp_factor = 0.05  # Slight bend
            elif severity == 'medium':
                warp_factor = 0.1
            else:  # heavy
                warp_factor = 0.2  # Significant bend
            
            # Create a bent/warped sign effect
            # Ensure valid bbox coordinates
            ymin = max(0, bbox['ymin'])
            ymax = min(image.shape[0], bbox['ymax'])
            xmin = max(0, bbox['xmin'])
            xmax = min(image.shape[1], bbox['xmax'])
            
            # Safety check to ensure sign area is not empty
            if ymin >= ymax or xmin >= xmax:
                # Just return the unmodified image if we can't extract a valid sign area
                print("Warning: Invalid sign area for bend effect")
                return image
                
            sign_area = image[ymin:ymax, xmin:xmax].copy()
            
            # Double check sign_area dimensions
            if sign_area.size == 0 or sign_area is None:
                print("Warning: Empty sign area for bend effect")
                return image
                
            # Create a warp/bend
            rows, cols = sign_area.shape[:2]
            
            # Additional check for minimum size
            if rows <= 2 or cols <= 2:
                print("Warning: Sign area too small for bend effect")
                return image
                
            # Randomly choose warp direction
            warp_direction = random.choice(['top', 'bottom', 'left', 'right'])
            
            # Create the appropriate warp with severity
            if warp_direction == 'top':
                bend = np.float32([
                    [0, 0], [cols-1, 0], 
                    [cols*warp_factor, rows-1], [cols*(1-warp_factor), rows-1]
                ])
            elif warp_direction == 'bottom':
                bend = np.float32([
                    [cols*warp_factor, 0], [cols*(1-warp_factor), 0], 
                    [0, rows-1], [cols-1, rows-1]
                ])
            elif warp_direction == 'left':
                bend = np.float32([
                    [0, 0], [cols-1, rows*warp_factor], 
                    [0, rows-1], [cols-1, rows*(1-warp_factor)]
                ])
            else:  # right
                bend = np.float32([
                    [0, rows*warp_factor], [cols-1, 0], 
                    [0, rows*(1-warp_factor)], [cols-1, rows-1]
                ])
            
            # Original corners
            original = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
            
            try:
                # Apply warp with safety checks
                M = cv2.getPerspectiveTransform(original, bend)
                
                # Verify sign_area is not empty before warping
                if sign_area.size > 0 and cols > 0 and rows > 0:
                    # Verify source points are within bounds
                    if np.all(original[:, 0] >= 0) and np.all(original[:, 0] < cols) and \
                       np.all(original[:, 1] >= 0) and np.all(original[:, 1] < rows):
                        warped_sign = cv2.warpPerspective(sign_area, M, (cols, rows))
                        
                        # Additional check before replacement
                        if warped_sign.shape == sign_area.shape and warped_sign.size > 0:
                            # Replace sign area with warped version
                            image[ymin:ymax, xmin:xmax] = warped_sign
                    else:
                        print("Warning: Source points out of bounds for warpPerspective")
                else:
                    print("Warning: Empty source for warpPerspective")
            except Exception as e:
                print(f"Warning: Error in warpPerspective: {e}")
        
        else:  # fade
            # Define parameters based on severity
            if severity == 'light':
                fade_spots = random.randint(2, 4)  # Fewer faded areas
                fade_intensity = 0.6  # Less fading
            elif severity == 'medium':
                fade_spots = random.randint(3, 8)
                fade_intensity = 0.4
            else:  # heavy
                fade_spots = random.randint(6, 12)  # More faded areas
                fade_intensity = 0.2  # More intense fading
            
            # Make parts of the sign fade/worn out
            try:
                # Get sign bounds with additional safety checks
                ymin, ymax = bbox['ymin'], bbox['ymax']
                xmin, xmax = bbox['xmin'], bbox['xmax']
                
                # Validate coordinates
                if ymin >= ymax or xmin >= xmax or ymin < 0 or xmin < 0 or ymax > image.shape[0] or xmax > image.shape[1]:
                    print("Warning: Invalid bounds for sign area in fade effect")
                    return image
                
                sign_area = image[ymin:ymax, xmin:xmax]
                
                # Verify sign_area is valid
                if sign_area.size == 0 or sign_area is None:
                    print("Warning: Empty sign area for fade effect")
                    return image
                
                # Create a fade mask
                fade_mask = np.ones_like(sign_area) * 255
            except Exception as e:
                print(f"Warning: Error setting up fade effect: {e}")
                return image
            
            # Add random faded areas
            for _ in range(fade_spots):
                fx = random.randint(0, sign_width-1)
                fy = random.randint(0, sign_height-1)
                fr = random.randint(sign_width//15, sign_width//6)
                
                # Create gradient fade spots with severity-based intensity
                for y in range(sign_height):
                    for x in range(sign_width):
                        dist = np.sqrt((x - fx)**2 + (y - fy)**2)
                        if dist < fr:
                            fade_factor = fade_intensity + (1.0 - fade_intensity) * (dist / fr)
                            for c in range(3):
                                if 0 <= y < fade_mask.shape[0] and 0 <= x < fade_mask.shape[1]:
                                    fade_mask[y, x, c] *= fade_factor
            
            # Apply fade mask (convert both to float32 for consistency)
            sign_area_float = sign_area.astype(np.float32)
            fade_mask_float = fade_mask.astype(np.float32)/255
            faded_sign = cv2.multiply(sign_area_float, fade_mask_float)
            image[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] = np.clip(faded_sign, 0, 255).astype(np.uint8)
            
        return image
    
    def _apply_fog_effect(self, image, severity='medium'):
        """
        Add a fog effect to an image with varying severity levels.
        severity: 'light', 'medium', or 'heavy'
        """
        image_float = image.astype(np.float32) / 255.0
        height, width = image_float.shape[:2]
        
        # Define parameters based on severity
        if severity == 'light':
            fog_base = 0.85  # Lighter fog
            fog_range = (0.8, 0.9)
            alpha = 0.25  # Lower intensity
            blur = 3
        elif severity == 'medium':
            fog_base = 0.8
            fog_range = (0.7, 0.9)
            alpha = 0.4
            blur = 5
        else:  # heavy
            fog_base = 0.75  # Darker fog
            fog_range = (0.65, 0.85)
            alpha = 0.6  # Higher intensity
            blur = 9  # More blur
        
        # Create a fog layer
        fog_layer = np.ones((height, width, 3), dtype=np.float32) * fog_base
        
        # Apply some random variation to make it more natural
        noise = np.random.normal(0, 0.05, (height, width, 3))
        fog_layer += noise
        fog_layer = np.clip(fog_layer, fog_range[0], fog_range[1])
        
        # Blend the fog with the image
        blended = image_float * (1 - alpha) + fog_layer * alpha
        
        # Ensure values are in valid range
        blended = np.clip(blended, 0, 1.0)
        
        # Convert back to uint8
        result = (blended * 255.0).astype(np.uint8)
        
        # Apply a blur to simulate fog scatter
        result = cv2.GaussianBlur(result, (blur, blur), 0)
        
        return result
        
    def _apply_shadow(self, image, severity='medium'):
        """
        Apply random shadow effect to image with varying severity levels.
        severity: 'light', 'medium', or 'heavy'
        """
        # Convert to float32 at the beginning
        image_float = image.astype(np.float32)
        height, width = image_float.shape[:2]
        
        # Define parameters based on severity
        if severity == 'light':
            shadow_intensity_range = (0.7, 0.9)  # Lighter shadow
            blur_size = 11  # Less blur on edges
            num_points_range = (3, 5)  # Simpler shadow shape
        elif severity == 'medium':
            shadow_intensity_range = (0.4, 0.7)  # Medium shadow
            blur_size = 21
            num_points_range = (3, 6)
        else:  # heavy
            shadow_intensity_range = (0.2, 0.5)  # Darker shadow
            blur_size = 31  # More blur for softer edges
            num_points_range = (4, 8)  # More complex shadow shape
        
        # Create random shadow polygon
        num_points = random.randint(num_points_range[0], num_points_range[1])
        points = []
        
        for _ in range(num_points):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            points.append([x, y])
        
        # Create shadow mask (as float)
        shadow_mask = np.zeros_like(image_float)
        points = np.array(points, np.int32)
        cv2.fillPoly(shadow_mask, [points], (1.0, 1.0, 1.0))
        
        # Blur shadow edges
        shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_size, blur_size), 0)
        
        # Apply shadow (darken image where mask is)
        shadow_intensity = random.uniform(shadow_intensity_range[0], shadow_intensity_range[1])
        shadow = image_float * shadow_intensity
        
        # Blend shadow based on mask (all inputs are float32)
        result = image_float * (1.0 - shadow_mask) + shadow * shadow_mask
        
        # Convert back to uint8 only at the end
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def validate_xml_annotation(self, xml_path, image_path):
        """
        Comprehensive validation of XML annotation file (enhanced from Step 3).
        Returns tuple: (is_valid, error_messages, warning_messages)
        """
        error_messages = []
        warning_messages = []
        
        try:
            # Check if file exists and is not empty
            if not os.path.exists(xml_path):
                return False, ["XML file does not exist"], []
            
            if os.path.getsize(xml_path) == 0:
                return False, ["XML file is empty"], []
            
            # Parse XML
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except ET.ParseError as e:
                return False, [f"XML parsing error: {str(e)}"], []
            
            # XML structure validation: Validates required XML elements exist
            required_elements = ['filename', 'size']
            for elem_name in required_elements:
                elem = root.find(elem_name)
                if elem is None:
                    warning_messages.append(f"Missing recommended XML element: {elem_name}")
            
            # Get image dimensions if image exists
            image_width = None
            image_height = None
            if os.path.exists(image_path):
                try:
                    import cv2
                    img = cv2.imread(image_path)
                    if img is not None:
                        image_height, image_width = img.shape[:2]
                except ImportError:
                    warning_messages.append("OpenCV not available for image dimension validation")
                except Exception as e:
                    warning_messages.append(f"Could not read image dimensions: {str(e)}")

            # Validate size element if present
            size_elem = root.find('size')
            if size_elem is not None:
                width_elem = size_elem.find('width')
                height_elem = size_elem.find('height')
                if width_elem is not None and height_elem is not None:
                    try:
                        xml_width = int(width_elem.text)
                        xml_height = int(height_elem.text)
                        # Compare with actual image dimensions if available
                        if image_width is not None and image_height is not None:
                            if xml_width != image_width or xml_height != image_height:
                                warning_messages.append(f"XML dimensions ({xml_width}x{xml_height}) don't match image dimensions ({image_width}x{image_height})")
                    except (ValueError, TypeError):
                        warning_messages.append("Invalid width/height values in XML size element")
            
            # Validate objects
            objects = root.findall('object')
            if len(objects) == 0:
                warning_messages.append("No objects found in annotation")
            
            for obj_idx, obj in enumerate(objects):
                # Validate class name (matching Step 3 validation)
                class_name_elem = obj.find('name')
                if class_name_elem is None:
                    class_name_elem = obj.find('n')  # Alternative tag
                
                if class_name_elem is None or class_name_elem.text is None:
                    error_messages.append(f"Object {obj_idx + 1}: Missing class name")
                    continue
                
                class_name = class_name_elem.text.strip()
                if not class_name:
                    error_messages.append(f"Object {obj_idx + 1}: Empty class name")
                    continue
                
                # Class name validation: Check if class name is valid (non-empty, proper format)
                if len(class_name.split()) > 1:
                    warning_messages.append(f"Object {obj_idx + 1}: Class name contains spaces: '{class_name}'")
                
                # Data type validation for class name (ensure it's a string)
                if not isinstance(class_name, str):
                    error_messages.append(f"Object {obj_idx + 1}: Class name is not a string: {type(class_name)}")
                
                # Validate bounding box
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    error_messages.append(f"Object {obj_idx + 1}: Missing bounding box")
                    continue
                
                # Check for required coordinates
                required_coords = ['xmin', 'ymin', 'xmax', 'ymax']
                coords = {}
                
                for coord_name in required_coords:
                    coord_elem = bndbox.find(coord_name)
                    if coord_elem is None or coord_elem.text is None:
                        error_messages.append(f"Object {obj_idx + 1}: Missing {coord_name} coordinate")
                        continue
                    
                    try:
                        # Data type validation: Ensures all numeric values can be properly converted
                        coord_str = coord_elem.text.strip()
                        if not coord_str:
                            error_messages.append(f"Object {obj_idx + 1}: Empty {coord_name} coordinate")
                            continue
                        
                        coord_value = float(coord_str)
                        coords[coord_name] = coord_value
                        
                        # Additional validation: Check for NaN or infinite values
                        if not (coord_value == coord_value):  # NaN check
                            error_messages.append(f"Object {obj_idx + 1}: {coord_name} is NaN")
                            continue
                        
                        if coord_value == float('inf') or coord_value == float('-inf'):
                            error_messages.append(f"Object {obj_idx + 1}: {coord_name} is infinite: {coord_value}")
                            continue
                            
                    except ValueError:
                        error_messages.append(f"Object {obj_idx + 1}: Invalid {coord_name} value (not numeric): '{coord_elem.text}'")
                        continue
                
                # If we have all coordinates, validate them
                if len(coords) == 4:
                    xmin, ymin, xmax, ymax = coords['xmin'], coords['ymin'], coords['xmax'], coords['ymax']
                    
                    # Check for negative values
                    if any(coord < 0 for coord in coords.values()):
                        error_messages.append(f"Object {obj_idx + 1}: Negative coordinates found")
                    
                    # Check coordinate order
                    if xmin >= xmax:
                        error_messages.append(f"Object {obj_idx + 1}: xmin ({xmin}) >= xmax ({xmax})")
                    
                    if ymin >= ymax:
                        error_messages.append(f"Object {obj_idx + 1}: ymin ({ymin}) >= ymax ({ymax})")
                    
                    # Check against image dimensions if available
                    if image_width and image_height:
                        if xmax > image_width:
                            error_messages.append(f"Object {obj_idx + 1}: xmax ({xmax}) exceeds image width ({image_width})")
                        
                        if ymax > image_height:
                            error_messages.append(f"Object {obj_idx + 1}: ymax ({ymax}) exceeds image height ({image_height})")
                    
                    # Check for unreasonably small bounding boxes
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    # Zero-size prevention: Prevents width or height of 0 (invisible bounding boxes)
                    if width == 0 or height == 0:
                        error_messages.append(f"Object {obj_idx + 1}: Zero-size bounding box (width={width:.1f}, height={height:.1f})")
                    
                    # Size validation with image-relative thresholds (matching Step 3)
                    if image_width and image_height:
                        width_ratio = width / image_width
                        height_ratio = height / image_height
                        
                        # Unreasonably small (< 0.1% of image dimension)
                        min_ratio = 0.001  # 0.1% like Step 3
                        if width_ratio < min_ratio or height_ratio < min_ratio:
                            error_messages.append(f"Object {obj_idx + 1}: Bounding box too small (width={width_ratio:.3%}, height={height_ratio:.3%}), minimum={min_ratio:.1%}")
                        
                        # Unreasonably large (> 95% of image) - matching Step 3
                        max_ratio = 0.95  # 95% like Step 3
                        if width_ratio > max_ratio or height_ratio > max_ratio:
                            warning_messages.append(f"Object {obj_idx + 1}: Very large bounding box (covers {width_ratio:.1%} x {height_ratio:.1%} of image)")
                    else:
                        # Fallback pixel-based validation when image dimensions unavailable
                        if width < 5 or height < 5:
                            warning_messages.append(f"Object {obj_idx + 1}: Very small bounding box (width={width:.1f}, height={height:.1f} pixels)")
            
            is_valid = len(error_messages) == 0
            return is_valid, error_messages, warning_messages
            
        except Exception as e:
            return False, [f"Validation error: {str(e)}"], []

    def correct_bbox_coordinates(self, bbox, img_shape):
        """
        Automatically correct minor floating-point precision errors in bounding box coordinates.
        This prevents issues like coordinates being -0.000001 or 1.000001 due to augmentation transforms.
        Returns corrected bbox dictionary.
        """
        try:
            height, width = img_shape[:2]
            
            # Create a copy to avoid modifying the original
            corrected_bbox = bbox.copy()
            
            # Get coordinates and ensure they're floats
            xmin = float(bbox.get('xmin', 0))
            ymin = float(bbox.get('ymin', 0))
            xmax = float(bbox.get('xmax', 0))
            ymax = float(bbox.get('ymax', 0))
            
            # Define tolerance for floating-point precision errors
            # Increased tolerance to handle augmentation transform precision issues
            tolerance = 1e-5  # Small tolerance for precision errors (0.00001)
            
            # Correct coordinates that are slightly out of bounds due to floating-point precision
            # For pixel coordinates (absolute values)
            if xmin < 0 and xmin >= -tolerance:
                xmin = 0.0
            if ymin < 0 and ymin >= -tolerance:
                ymin = 0.0
            if xmax > width and xmax <= width + tolerance:
                xmax = float(width)
            if ymax > height and ymax <= height + tolerance:
                ymax = float(height)
                
            # For normalized coordinates (0-1 range) - check if coordinates seem normalized
            if all(0 <= coord <= 1.1 for coord in [xmin, ymin, xmax, ymax]):  # Allow slight overshoot to detect normalized coords
                if xmin < 0 and xmin >= -tolerance:
                    xmin = 0.0
                if ymin < 0 and ymin >= -tolerance:
                    ymin = 0.0
                if xmax > 1.0 and xmax <= 1.0 + tolerance:
                    xmax = 1.0
                if ymax > 1.0 and ymax <= 1.0 + tolerance:
                    ymax = 1.0
            
            # Ensure proper coordinate order (min < max)
            if xmin >= xmax:
                # Swap if they're very close (precision error)
                if abs(xmin - xmax) <= tolerance:
                    xmin, xmax = min(xmin, xmax), max(xmin, xmax) + tolerance
            
            if ymin >= ymax:
                # Swap if they're very close (precision error)
                if abs(ymin - ymax) <= tolerance:
                    ymin, ymax = min(ymin, ymax), max(ymin, ymax) + tolerance
            
            # Update the corrected bbox
            corrected_bbox.update({
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
            
            return corrected_bbox
            
        except Exception as e:
            print(f"Warning: Error correcting bbox coordinates: {e}")
            return bbox  # Return original if correction fails

    def validate_for_yolo_conversion(self, bbox, img_shape, filename):
        """
        Strict validation specifically designed to prevent Step 5 YOLO conversion errors.
        This catches the exact issues that cause "extends above/below/left/right of image" errors.
        Returns tuple: (is_valid, error_messages, warning_messages)
        """
        error_messages = []
        warning_messages = []
        
        try:
            height, width = img_shape[:2]
            
            # Get coordinates
            xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
            
            # Convert to floats for precise checking
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
            
            # STRICT boundary checks - exactly what Step 5 will validate
            # These are the exact conditions that cause Step 5 errors
            
            # Check for coordinates at or beyond boundaries (the root cause of Step 5 errors)
            tolerance = 1e-10  # Very strict tolerance for Step 5 compatibility
            
            if xmin <= tolerance:
                error_messages.append(f"xmin too close to left boundary: {xmin} (will cause 'extends left of image' error)")
            
            if ymin <= tolerance:
                error_messages.append(f"ymin too close to top boundary: {ymin} (will cause 'extends above image' error)")
                
            if xmax >= (width - tolerance):
                error_messages.append(f"xmax too close to right boundary: {xmax} >= {width} (will cause 'extends right of image' error)")
                
            if ymax >= (height - tolerance):
                error_messages.append(f"ymax too close to bottom boundary: {ymax} >= {height} (will cause 'extends below image' error)")
            
            # Check for normalized coordinate issues (if this might be normalized)
            if all(0 <= coord <= 1.1 for coord in [xmin, ymin, xmax, ymax]):
                if xmin <= tolerance:
                    error_messages.append(f"Normalized xmin too close to 0: {xmin}")
                if ymin <= tolerance:
                    error_messages.append(f"Normalized ymin too close to 0: {ymin}")
                if xmax >= (1.0 - tolerance):
                    error_messages.append(f"Normalized xmax too close to 1: {xmax}")
                if ymax >= (1.0 - tolerance):
                    error_messages.append(f"Normalized ymax too close to 1: {ymax}")
            
            # Additional validations
            if xmin >= xmax:
                error_messages.append(f"Invalid coordinate order: xmin ({xmin}) >= xmax ({xmax})")
            if ymin >= ymax:
                error_messages.append(f"Invalid coordinate order: ymin ({ymin}) >= ymax ({ymax})")
                
            # Check for reasonable bounding box size
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            
            if bbox_width <= 1.0:
                error_messages.append(f"Bounding box too narrow: width={bbox_width}")
            if bbox_height <= 1.0:
                error_messages.append(f"Bounding box too short: height={bbox_height}")
            
            is_valid = len(error_messages) == 0
            return is_valid, error_messages, warning_messages
            
        except Exception as e:
            error_messages.append(f"Validation error: {str(e)}")
            return False, error_messages, warning_messages

    def validate_augmented_bbox(self, bbox, img_shape, filename):
        """
        Comprehensive validation of augmented bounding box using same standards as Step 3 and Step 5.
        Returns tuple: (is_valid, error_messages, warning_messages)
        """
        error_messages = []
        warning_messages = []
        
        try:
            height, width = img_shape[:2]
            
            # Check if all required bbox coordinates are present
            required_keys = ['xmin', 'ymin', 'xmax', 'ymax', 'class']
            for key in required_keys:
                if key not in bbox:
                    error_messages.append(f"Missing required bbox key: {key}")
                    return False, error_messages, warning_messages
            
            # Validate coordinate values
            xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
            
            # Data type validation: Ensures all numeric values can be properly converted
            try:
                xmin = float(xmin)
                ymin = float(ymin)
                xmax = float(xmax)
                ymax = float(ymax)
            except (ValueError, TypeError):
                error_messages.append(f"Invalid coordinate values (not numeric): xmin={bbox['xmin']}, ymin={bbox['ymin']}, xmax={bbox['xmax']}, ymax={bbox['ymax']}")
                return False, error_messages, warning_messages
            
            # Check for NaN or infinite values
            coords = [xmin, ymin, xmax, ymax]
            for i, coord in enumerate(coords):
                coord_names = ['xmin', 'ymin', 'xmax', 'ymax']
                if not (coord == coord):  # NaN check
                    error_messages.append(f"{coord_names[i]} is NaN")
                if coord == float('inf') or coord == float('-inf'):
                    error_messages.append(f"{coord_names[i]} is infinite: {coord}")
            
            # Check for negative values
            if any(coord < 0 for coord in coords):
                error_messages.append(f"Negative coordinates found: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
            
            # Check coordinate order
            if xmin >= xmax:
                error_messages.append(f"xmin ({xmin}) >= xmax ({xmax})")
            
            if ymin >= ymax:
                error_messages.append(f"ymin ({ymin}) >= ymax ({ymax})")
            
            # Check against image dimensions
            if xmax > width:
                error_messages.append(f"xmax ({xmax}) exceeds image width ({width})")
            
            if ymax > height:
                error_messages.append(f"ymax ({ymax}) exceeds image height ({height})")
            
            # Check for unreasonably small bounding boxes
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            
            # Zero-size prevention: Prevents width or height of 0 (invisible bounding boxes)
            if bbox_width == 0 or bbox_height == 0:
                error_messages.append(f"Zero-size bounding box (width={bbox_width:.1f}, height={bbox_height:.1f})")
            
            # Size validation with image-relative thresholds (matching Step 3)
            if width > 0 and height > 0:
                width_ratio = bbox_width / width
                height_ratio = bbox_height / height
                
                # Unreasonably small (< 0.1% of image dimension)
                min_ratio = 0.001  # 0.1% like Step 3
                if width_ratio < min_ratio or height_ratio < min_ratio:
                    error_messages.append(f"Bounding box too small (width={width_ratio:.3%}, height={height_ratio:.3%}), minimum={min_ratio:.1%}")
                
                # Unreasonably large (> 95% of image) - matching Step 3
                max_ratio = 0.95  # 95% like Step 3
                if width_ratio > max_ratio or height_ratio > max_ratio:
                    warning_messages.append(f"Very large bounding box (covers {width_ratio:.1%} x {height_ratio:.1%} of image)")
            else:
                # Fallback pixel-based validation when image dimensions unavailable
                if bbox_width < 5 or bbox_height < 5:
                    warning_messages.append(f"Very small bounding box (width={bbox_width:.1f}, height={bbox_height:.1f} pixels)")
            
            # Class name validation (matching Step 3)
            class_name = bbox.get('class', '')
            if not class_name or not isinstance(class_name, str):
                error_messages.append(f"Invalid or missing class name: '{class_name}'")
            elif len(class_name.split()) > 1:
                warning_messages.append(f"Class name contains spaces: '{class_name}'")
            
            is_valid = len(error_messages) == 0
            return is_valid, error_messages, warning_messages
            
        except Exception as e:
            return False, [f"Validation error: {str(e)}"], []
    
    def _save_augmented_annotation(self, src_xml_path, dst_xml_path, bbox, filename, img_shape):
        """Save augmented annotation XML with updated bbox and comprehensive validation."""
        try:
            # First, correct any minor floating-point precision errors in bbox coordinates
            corrected_bbox = self.correct_bbox_coordinates(bbox, img_shape)
            
            # STRICT validation specifically for Step 5 YOLO conversion compatibility
            is_yolo_valid, yolo_errors, yolo_warnings = self.validate_for_yolo_conversion(corrected_bbox, img_shape, filename)
            
            if not is_yolo_valid:
                print(f"❌ REJECTING augmented annotation {filename} - would cause Step 5 YOLO errors:")
                for error in yolo_errors:
                    print(f"   - YOLO ERROR: {error}")
                print(f"   This augmentation will be skipped to prevent Step 5 validation failures.")
                return False
            
            # Additional comprehensive validation (using same standards as Step 3 and Step 5)
            is_valid, errors, warnings = self.validate_augmented_bbox(corrected_bbox, img_shape, filename)
            
            if not is_valid:
                print(f"❌ Validation failed for augmented annotation {filename}:")
                for error in errors:
                    print(f"   - ERROR: {error}")
                for warning in warnings:
                    print(f"   - WARNING: {warning}")
                print(f"   Skipping this augmented annotation to maintain data quality.")
                return False
            
            if warnings:
                print(f"⚠️ Validation warnings for augmented annotation {filename}:")
                for warning in warnings:
                    print(f"   - WARNING: {warning}")
            
            # Parse the original XML
            tree = ET.parse(src_xml_path)
            root = tree.getroot()
            
            # Update filename
            filename_elem = root.find('filename')
            if filename_elem is not None:
                filename_elem.text = filename
            
            # Update image size
            size_elem = root.find('size')
            if size_elem is not None:
                width_elem = size_elem.find('width')
                height_elem = size_elem.find('height')
                depth_elem = size_elem.find('depth')
                
                if width_elem is not None:
                    width_elem.text = str(img_shape[1])
                if height_elem is not None:
                    height_elem.text = str(img_shape[0])
                if depth_elem is not None and len(img_shape) > 2:
                    depth_elem.text = str(img_shape[2])
            
            # Update bounding box with validated coordinates
            for obj in root.findall('object'):
                # Validate class name matches
                class_name_elem = obj.find('name')
                if class_name_elem is None:
                    class_name_elem = obj.find('n')  # Alternative tag
                
                if class_name_elem is not None:
                    # Ensure class name consistency
                    class_name_elem.text = corrected_bbox['class']
                
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    # Update with corrected and validated coordinates (ensure they're integers for XML)
                    xmin_elem = bndbox.find('xmin')
                    ymin_elem = bndbox.find('ymin')
                    xmax_elem = bndbox.find('xmax')
                    ymax_elem = bndbox.find('ymax')
                    
                    if xmin_elem is not None:
                        xmin_elem.text = str(int(corrected_bbox['xmin']))
                    if ymin_elem is not None:
                        ymin_elem.text = str(int(corrected_bbox['ymin']))
                    if xmax_elem is not None:
                        xmax_elem.text = str(int(corrected_bbox['xmax']))
                    if ymax_elem is not None:
                        ymax_elem.text = str(int(corrected_bbox['ymax']))
            
            # Save the updated XML with proper encoding
            tree.write(dst_xml_path, encoding='utf-8', xml_declaration=True)
            
            # Final validation: Re-read and validate the saved file
            try:
                # Quick validation of saved file
                saved_tree = ET.parse(dst_xml_path)
                saved_root = saved_tree.getroot()
                
                # Check that we can find the object and bounding box
                saved_obj = saved_root.find('object')
                if saved_obj is None:
                    print(f"⚠️ Warning: No object found in saved XML {dst_xml_path}")
                    return False
                
                saved_bndbox = saved_obj.find('bndbox')
                if saved_bndbox is None:
                    print(f"⚠️ Warning: No bounding box found in saved XML {dst_xml_path}")
                    return False
                
                return True
                
            except Exception as e:
                print(f"❌ Error validating saved XML {dst_xml_path}: {str(e)}")
                return False
            
        except Exception as e:
            print(f"❌ Error saving augmented annotation {dst_xml_path}: {str(e)}")
            return False
    
    def view_augmentation_stats(self):
        """Display statistics about augmentation including severity levels."""
        print("\n" + "="*60)
        print("           📊  AUGMENTATION STATISTICS  📊")
        print("="*60)
        
        if not self.augmentation_data.get("classes"):
            print("\n❌ No augmentation has been performed yet.")
            return
        
        total_original = 0
        total_augmented = 0
        total_severity = {"light": 0, "medium": 0, "heavy": 0}
        total_conditions = {}
        
        print(f"\n📋 Class-wise Augmentation Stats:")
        for sequential_id, data in self.augmentation_data["classes"].items():
            class_name = data["class_name"]
            original_count = data["original_image_count"]
            augmented_count = data["augmented_count"] if "augmented_count" in data else data["augmented_image_count"]
            
            total_original += original_count
            total_augmented += augmented_count
            
            print(f"   - {sequential_id}_{class_name}: {original_count} original → {augmented_count} augmented")
            
            # Show severity distribution if available
            if "severity_stats" in data:
                severity_stats = data["severity_stats"]
                print(f"     Severity: Light: {severity_stats.get('light', 0)}, "
                      f"Medium: {severity_stats.get('medium', 0)}, "
                      f"Heavy: {severity_stats.get('heavy', 0)}")
                
                # Add to totals
                for severity, count in severity_stats.items():
                    total_severity[severity] += count
            
            # Show condition types if available
            if "condition_stats" in data:
                condition_stats = data["condition_stats"]
                conditions_str = ", ".join([f"{cond}: {count}" for cond, count in sorted(
                    condition_stats.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )])
                print(f"     Conditions: {conditions_str}")
                
                # Add to totals
                for condition, count in condition_stats.items():
                    if condition not in total_conditions:
                        total_conditions[condition] = 0
                    total_conditions[condition] += count
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total Original Images: {total_original}")
        print(f"   Total Augmented Images: {total_augmented}")
        print(f"   Total Dataset Size: {total_original + total_augmented}")
        print(f"   Expansion Factor: {(total_original + total_augmented) / total_original:.1f}x")
        
        # Show overall severity distribution
        if total_severity["light"] + total_severity["medium"] + total_severity["heavy"] > 0:
            print(f"\n🔆 Severity Distribution:")
            for severity, count in total_severity.items():
                percentage = 100 * count / total_augmented if total_augmented > 0 else 0
                print(f"   - {severity.capitalize()}: {count} images ({percentage:.1f}%)")
        
        # Show overall condition distribution
        if total_conditions:
            print(f"\n🌈 Condition Distribution:")
            for condition, count in sorted(total_conditions.items(), key=lambda x: x[1], reverse=True):
                percentage = 100 * count / sum(total_conditions.values())
                print(f"   - {condition.capitalize()}: {count} applications ({percentage:.1f}%)")
    
    def clear_all_augmented_files(self):
        """Clear all augmented files from the entire dataset."""
        print("\n=== Clearing All Existing Augmented Files ONLY ===")
        print("⚠️ NOTE: Original images and annotations will NOT be touched")
        
        total_images_deleted = 0
        total_annotations_deleted = 0
        
        # Check both mandatory and cautionary directories
        for category in ['Mandatory_Road_Signs', 'Cautionary_Road_Signs']:
            category_dir = os.path.join(self.organized_dataset_dir, category)
            
            if not os.path.exists(category_dir):
                continue
                
            # Process each class folder
            for class_folder in os.listdir(category_dir):
                class_path = os.path.join(category_dir, class_folder)
                
                if not os.path.isdir(class_path):
                    continue
                
                # SAFETY CHECK: Make sure this is a class folder (should have images and annotations directories)
                if not os.path.exists(os.path.join(class_path, 'images')) or not os.path.exists(os.path.join(class_path, 'annotations')):
                    print(f"   ⚠️ Skipping {class_folder} - not a standard class folder")
                    continue
                    
                # Check for augmented directories - these should be named EXACTLY 'augmented_images' and 'augmented_annotations'
                augmented_images_dir = os.path.join(class_path, 'augmented_images')
                augmented_annotations_dir = os.path.join(class_path, 'augmented_annotations')
                
                # Clean up augmented images - ONLY in the augmented_images directory
                if os.path.exists(augmented_images_dir) and os.path.basename(augmented_images_dir) == 'augmented_images':
                    files = [f for f in os.listdir(augmented_images_dir) 
                             if os.path.isfile(os.path.join(augmented_images_dir, f)) and 
                             f.endswith(('.jpg', '.jpeg', '.png'))]
                    
                    files_count = len(files)
                    if files_count > 0:
                        for file in files:
                            file_path = os.path.join(augmented_images_dir, file)
                            try:
                                os.remove(file_path)
                                total_images_deleted += 1
                            except Exception as e:
                                print(f"   ⚠️ Error deleting {file}: {e}")
                        print(f"   🧹 Removed {files_count} augmented images from {class_folder}")
                
                # Clean up augmented annotations - ONLY in the augmented_annotations directory
                if os.path.exists(augmented_annotations_dir) and os.path.basename(augmented_annotations_dir) == 'augmented_annotations':
                    files = [f for f in os.listdir(augmented_annotations_dir) 
                             if os.path.isfile(os.path.join(augmented_annotations_dir, f)) and 
                             f.endswith('.xml')]
                    
                    files_count = len(files)
                    if files_count > 0:
                        for file in files:
                            file_path = os.path.join(augmented_annotations_dir, file)
                            try:
                                os.remove(file_path)
                                total_annotations_deleted += 1
                            except Exception as e:
                                print(f"   ⚠️ Error deleting {file}: {e}")
                        print(f"   🧹 Removed {files_count} augmented annotations from {class_folder}")
        
        # Reset augmentation data
        if "classes" in self.augmentation_data:
            self.augmentation_data["classes"] = {}
        if "stats" in self.augmentation_data:
            self.augmentation_data["stats"]["total_augmented_images"] = 0
            self.augmentation_data["stats"]["total_images_after_augmentation"] = self.progress_data['completed_images']
        
        # Save reset config
        self.save_augmentation_config()
        
        print(f"\n✅ Cleanup Summary:")
        print(f"   - {total_images_deleted} augmented images deleted")
        print(f"   - {total_annotations_deleted} augmented annotations deleted")
        print(f"   - 0 original files affected (preserved)")
        
        return True
            
    def run(self):
        """Run the augmentation process."""
        if not self.load_configuration():
            return False
        
        if not self.load_progress_data():
            return False
        
        if not self.create_augmentation_config():
            return False
        
        # Clear all existing augmented files before starting new augmentation
        print("\n⚠️ This will delete all existing augmented files and start fresh.")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            self.clear_all_augmented_files()
        else:
            print("❌ Augmentation cancelled.")
            return False
        
        classes_to_augment = self.get_classes_to_augment()
        
        if not classes_to_augment:
            print(f"\n❌ No labeled classes found. Please complete labeling in Step 3 first.")
            return False
        
        while True:
            try:
                choice = self.display_augmentation_menu(classes_to_augment)
                
                if choice == '1':
                    # Augment all classes
                    print(f"\n🔄 Starting augmentation for all classes...")
                    for class_info in classes_to_augment:
                        sequential_id = class_info['sequential_id']
                        if sequential_id not in self.augmentation_data.get("classes", {}):
                            self.augment_class(class_info)
                    print(f"\n🎉 All classes have been augmented!")
                    
                elif choice == '2':
                    # Augment specific class
                    class_idx = int(input(f"\n🎯 Enter class number (1-{len(classes_to_augment)}): ")) - 1
                    if 0 <= class_idx < len(classes_to_augment):
                        self.augment_class(classes_to_augment[class_idx])
                    else:
                        print(f"❌ Invalid class number")
                
                elif choice == '3':
                    # View stats
                    self.view_augmentation_stats()
                
                elif choice == '4':
                    # Exit
                    print(f"\n👋 Exiting Augmentation System")
                    break
                
                else:
                    print(f"❌ Invalid choice. Please enter 1-4.")
                
                input(f"\n⏳ Press Enter to continue...")
                
            except KeyboardInterrupt:
                print(f"\n\n🛑 Augmentation system interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                input(f"\n⏳ Press Enter to continue...")
        
        # Calculate final statistics
        total_original = sum(c["original_image_count"] for c in self.augmentation_data.get("classes", {}).values())
        total_augmented = sum(c.get("augmented_count", c.get("augmented_image_count", 0)) 
                             for c in self.augmentation_data.get("classes", {}).values())
        
        self.augmentation_data["stats"]["total_original_images"] = total_original
        self.augmentation_data["stats"]["total_augmented_images"] = total_augmented
        self.augmentation_data["stats"]["total_images_after_augmentation"] = total_original + total_augmented
        self.save_augmentation_config()
        
        print(f"\n🎯 Next Step: Run Step5_yolo_conversion.py to convert annotations to YOLO format")
        return True


def main():
    organized_dataset_dir = '/Users/jvarghese/Documents/TrafficSignProject/organized_dataset'
    
    print("=== Step 4: Image Augmentation ===")
    print("This process will create augmented versions of the labeled traffic sign images")
    print("and update their annotations accordingly.\n")
    
    augmenter = ImageAugmenter(organized_dataset_dir)
    augmenter.run()


if __name__ == "__main__":
    main()