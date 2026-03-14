#!/usr/bin/env python3
"""
Step 3: Batch Labeling System
Provides an efficient system for labeling all traffic sign images using labelImg.
Manages the labeling workflow across all classes with progress tracking.
"""

import os
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


class BatchLabelingManager:
    def __init__(self, config_file='traffic_sign_config.json'):
        """Initialize the batch labeling system."""
        self.config_file = config_file
        self.config_data = None
        self.validation_enabled = True  # Enable comprehensive validation by default
        self.backup_created = False
        # Progress file path (default in organized_dataset)
        self.progress_file = '/Users/jvarghese/Documents/TrafficSignProject/organized_dataset/batch_labeling_progress.json'
        # labelImg executable path - use virtual environment
        self.labelimg_path = '/Users/jvarghese/Documents/TrafficSignProject/.venv39/bin/labelImg'
        self.venv_python = '/Users/jvarghese/Documents/TrafficSignProject/.venv39/bin/python3.9'
        self.venv_pip = '/Users/jvarghese/Documents/TrafficSignProject/.venv39/bin/pip3.9'
        # Auto-fix settings
        self.auto_fix_issues = True
        
    def validate_xml_annotation(self, xml_path, image_path):
        """
        Comprehensive validation of XML annotation file.
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
                # Validate class name (matching Step 5 validation)
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
                    
                    # Size validation with image-relative thresholds (matching Step 5)
                    if image_width and image_height:
                        width_ratio = width / image_width
                        height_ratio = height / image_height
                        
                        # Unreasonably small (< 0.1% of image dimension)
                        min_ratio = 0.001  # 0.1% like Step 5
                        if width_ratio < min_ratio or height_ratio < min_ratio:
                            error_messages.append(f"Object {obj_idx + 1}: Bounding box too small (width={width_ratio:.3%}, height={height_ratio:.3%}), minimum={min_ratio:.1%}")
                        
                        # Unreasonably large (> 95% of image) - matching Step 5
                        max_ratio = 0.95  # 95% like Step 5
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
    
    def validate_class_annotations(self, class_info, show_details=True):
        """
        Validate all annotations for a specific class with detailed reporting.
        Returns summary of validation results.
        """
        images_dir = class_info['images_dir']
        annotations_dir = class_info['annotations_dir']
        
        if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
            return {'valid': 0, 'invalid': 0, 'missing': 0, 'total_errors': 0, 'total_warnings': 0}
        
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Get all XML files
        xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
        
        validation_results = {
            'valid': 0,
            'invalid': 0,
            'missing': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'error_details': [],
            'warning_details': [],
            'missing_files': [],
            'orphaned_files': []
        }
        
        if show_details:
            print(f"\n🔍 Validating annotations for {class_info['sequential_id']}_{class_info['class_name']}")
        
        # Check each image for corresponding annotation
        for img_file in image_files:
            img_base = os.path.splitext(img_file)[0]
            xml_file = img_base + '.xml'
            
            if xml_file in xml_files:
                # Validate the XML file
                img_path = os.path.join(images_dir, img_file)
                xml_path = os.path.join(annotations_dir, xml_file)
                
                is_valid, errors, warnings = self.validate_xml_annotation(xml_path, img_path)
                
                if is_valid:
                    validation_results['valid'] += 1
                else:
                    validation_results['invalid'] += 1
                    
                    # Count objects for statistics (matching Step 5 detailed reporting)
                    try:
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        objects = root.findall('object')
                        object_count = len(objects)
                    except:
                        object_count = 0
                    
                    validation_results['error_details'].append({
                        'file': xml_file,
                        'errors': errors,
                        'warnings': warnings,
                        'objects': list(range(object_count))  # For statistics tracking
                    })
                
                validation_results['total_errors'] += len(errors)
                validation_results['total_warnings'] += len(warnings)
                
                if show_details and (errors or warnings):
                    print(f"  📄 {xml_file}:")
                    # Error categorization: Distinguishes between errors (critical) and warnings (non-critical)
                    for error in errors:
                        print(f"    ❌ ERROR: {error}")
                    for warning in warnings:
                        print(f"    ⚠️ WARNING: {warning}")
                        
            else:
                validation_results['missing'] += 1
                validation_results['missing_files'].append(f"{img_file} (no XML annotation)")
                if show_details:
                    print(f"  📄 {img_file}: ❌ Missing annotation file ({xml_file})")
        
        # Check for orphaned XML files (File consistency: Checks for orphaned labels without images)
        image_bases = {os.path.splitext(img)[0] for img in image_files}
        for xml_file in xml_files:
            xml_base = os.path.splitext(xml_file)[0]
            if xml_base not in image_bases:
                validation_results['orphaned_files'].append(f"{xml_file} (no corresponding image)")
                if show_details:
                    print(f"  🗂️ {xml_file}: ⚠️ Orphaned annotation (no corresponding image)")
        
        if show_details:
            print(f"  📊 Summary: {validation_results['valid']} valid, {validation_results['invalid']} invalid, {validation_results['missing']} missing")
            
            # High-level validation status reporting
            print(f"  ")
            
            # File Integrity Status
            file_integrity_status = "✅" if validation_results['total_errors'] == 0 and validation_results['missing'] == 0 else "❌" if validation_results['total_errors'] > 0 else "⚠️"
            print(f"  {file_integrity_status} File Integrity Status:")
            
            missing_count = validation_results['missing']
            xml_existence = '✅ All files present' if missing_count == 0 else f'❌ {missing_count} missing'
            print(f"     • XML Existence: {xml_existence}")
            
            invalid_count = validation_results['invalid']
            xml_parsing = '✅ All parseable' if invalid_count == 0 else f'❌ {invalid_count} unparseable'
            print(f"     • XML Parsing: {xml_parsing}")
            
            file_size_status = '✅ No empty files' if validation_results['total_errors'] == 0 else '❌ Empty files detected'
            print(f"     • File Size: {file_size_status}")
            
            # Coordinate Validation Status
            coord_errors = sum(1 for detail in validation_results['error_details'] 
                              for error in detail['errors'] 
                              if any(coord in error.lower() for coord in ['coordinate', 'xmin', 'ymin', 'xmax', 'ymax', 'bounding box']))
            coord_status = "✅" if coord_errors == 0 else "❌"
            print(f"  ")
            print(f"  {coord_status} Coordinate Validation Status:")
            
            bbox_structure = '✅ All complete' if coord_errors == 0 else f'❌ {coord_errors} issues found'
            print(f"     • Bounding Box Structure: {bbox_structure}")
            
            coord_values = '✅ All numeric & valid' if coord_errors == 0 else '❌ Invalid values detected'
            print(f"     • Coordinate Values: {coord_values}")
            
            image_bounds = '✅ Within boundaries' if coord_errors == 0 else '❌ Boundary violations found'
            print(f"     • Image Bounds: {image_bounds}")
            
            size_requirements = '✅ Proper dimensions' if coord_errors == 0 else '❌ Size issues detected'
            print(f"     • Size Requirements: {size_requirements}")
            
            # Class Name Validation Status
            class_errors = sum(1 for detail in validation_results['error_details'] 
                              for error in detail['errors'] 
                              if 'class name' in error.lower())
            class_status = "✅" if class_errors == 0 else "❌"
            print(f"  ")
            print(f"  {class_status} Class Name Validation Status:")
            
            label_presence = '✅ All objects labeled' if class_errors == 0 else f'❌ {class_errors} missing labels'
            print(f"     • Label Presence: {label_presence}")
            
            label_format = '✅ Valid formatting' if class_errors == 0 else '❌ Format issues detected'
            print(f"     • Label Format: {label_format}")
            
            data_types = '✅ Proper string types' if class_errors == 0 else '❌ Type issues detected'
            print(f"     • Data Types: {data_types}")
            
            # File Consistency Status
            consistency_status = "✅" if len(validation_results['missing_files']) == 0 and len(validation_results['orphaned_files']) == 0 else "⚠️"
            print(f"  ")
            print(f"  {consistency_status} File Consistency Status:")
            
            pair_status = '✅' if validation_results['missing'] == 0 else '❌'
            print(f"     • Image-XML Pairs: {pair_status} {validation_results['valid']}/{len(image_files)} matched")
            
            orphaned_count = len(validation_results['orphaned_files'])
            orphaned_status = '✅ 0 orphaned' if orphaned_count == 0 else f'⚠️ {orphaned_count} orphaned'
            print(f"     • Orphaned Files: {orphaned_status}")
            
            missing_annotations = validation_results['missing']
            missing_status = '✅ 0 missing' if missing_annotations == 0 else f'❌ {missing_annotations} missing'
            print(f"     • Missing Annotations: {missing_status}")
            
            # Enhanced error reporting with categorization
            print(f"  ")
            if validation_results['total_errors'] > 0:
                print(f"  ❌ Critical errors: {validation_results['total_errors']} (will cause training failures)")
            
            if validation_results['total_warnings'] > 0:
                print(f"  ⚠️ Warnings: {validation_results['total_warnings']} (should be reviewed)")
            
            # Statistics reporting: Provides detailed counts
            total_objects = sum(len(detail.get('objects', [])) for detail in validation_results['error_details'])
            print(f"  📊 Detailed Statistics:")
            print(f"     • Image files: {len(image_files)}")
            print(f"     • XML files: {len(xml_files)}")
            print(f"     • Total objects detected: {total_objects if total_objects > 0 else 'N/A'}")
            print(f"     • Files with issues: {validation_results['invalid']}")
            print(f"     • Orphaned files: {len(validation_results['orphaned_files'])}")
            
            # Overall Status
            print(f"  ")
            if validation_results['total_errors'] == 0 and validation_results['missing'] == 0:
                if validation_results['total_warnings'] == 0:
                    print(f"  🟢 Overall Status: VALIDATION PASSED - Ready for training")
                else:
                    print(f"  🟡 Overall Status: VALIDATION PASSED with {validation_results['total_warnings']} warnings - Review recommended")
            else:
                print(f"  🔴 Overall Status: VALIDATION FAILED - {validation_results['total_errors']} critical errors, {validation_results['missing']} missing files")
        
        return validation_results
    
    def fix_annotation_issues(self, class_info, validation_results):
        """
        Attempt to automatically fix common annotation issues (matching Step 5 comprehensive fixes).
        """
        if not self.auto_fix_issues:
            return False
        
        fixed_count = 0
        annotations_dir = class_info['annotations_dir']
        images_dir = class_info['images_dir']
        
        print(f"\n🔧 Attempting to fix annotation issues...")
        
        # Remove orphaned XML files
        for orphaned_file in validation_results['orphaned_files']:
            try:
                # Remove the "(no corresponding image)" suffix if present
                clean_file = orphaned_file.split(' (')[0] if ' (' in orphaned_file else orphaned_file
                orphaned_path = os.path.join(annotations_dir, clean_file)
                if os.path.exists(orphaned_path):
                    os.remove(orphaned_path)
                    print(f"  🗑️ Removed orphaned file: {clean_file}")
                    fixed_count += 1
            except Exception as e:
                print(f"  ❌ Could not remove {clean_file}: {str(e)}")
        
        # Fix coordinate and XML issues in files with errors
        for error_detail in validation_results['error_details']:
            xml_file = error_detail['file']
            xml_path = os.path.join(annotations_dir, xml_file)
            
            if not os.path.exists(xml_path):
                continue
            
            try:
                # Load and parse XML
                tree = ET.parse(xml_path)
                root = tree.getroot()
                modified = False
                
                # Get image dimensions for coordinate validation
                img_file = xml_file.replace('.xml', '.jpg')
                img_path = os.path.join(images_dir, img_file)
                image_width = image_height = None
                
                if os.path.exists(img_path):
                    try:
                        import cv2
                        img = cv2.imread(img_path)
                        if img is not None:
                            image_height, image_width = img.shape[:2]
                    except:
                        pass
                
                # Fix issues in each object
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    if bndbox is None:
                        continue
                    
                    # Get coordinates
                    coords = {}
                    for coord_name in ['xmin', 'ymin', 'xmax', 'ymax']:
                        coord_elem = bndbox.find(coord_name)
                        if coord_elem is not None and coord_elem.text is not None:
                            try:
                                coords[coord_name] = float(coord_elem.text.strip())
                            except ValueError:
                                continue
                    
                    if len(coords) == 4:
                        xmin, ymin, xmax, ymax = coords['xmin'], coords['ymin'], coords['xmax'], coords['ymax']
                        original_coords = (xmin, ymin, xmax, ymax)
                        
                        # Fix negative coordinates (clip to 0)
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = max(0, xmax)
                        ymax = max(0, ymax)
                        
                        # Fix coordinates that exceed image boundaries
                        if image_width is not None and image_height is not None:
                            xmin = min(xmin, image_width - 1)
                            ymin = min(ymin, image_height - 1)
                            xmax = min(xmax, image_width)
                            ymax = min(ymax, image_height)
                        
                        # Fix invalid coordinate order (swap if needed)
                        if xmin >= xmax:
                            xmin, xmax = min(original_coords[0], original_coords[2]), max(original_coords[0], original_coords[2])
                        
                        if ymin >= ymax:
                            ymin, ymax = min(original_coords[1], original_coords[3]), max(original_coords[1], original_coords[3])
                        
                        # Ensure minimum bounding box size
                        if xmax - xmin < 1:
                            xmax = xmin + 1
                        if ymax - ymin < 1:
                            ymax = ymin + 1
                        
                        # Update coordinates if changed
                        if (xmin, ymin, xmax, ymax) != original_coords:
                            bndbox.find('xmin').text = str(int(xmin))
                            bndbox.find('ymin').text = str(int(ymin))
                            bndbox.find('xmax').text = str(int(xmax))
                            bndbox.find('ymax').text = str(int(ymax))
                            modified = True
                
                # Save if modified
                if modified:
                    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                    print(f"  ✅ Fixed coordinate issues in: {xml_file}")
                    fixed_count += 1
                    
            except Exception as e:
                print(f"  ❌ Could not fix {xml_file}: {str(e)}")
        
        if fixed_count > 0:
            print(f"  ✅ Fixed {fixed_count} issues automatically")
            return True
        else:
            print(f"  ℹ️ No issues could be automatically fixed")
            return False
    
    def toggle_validation(self):
        """Toggle comprehensive validation on/off."""
        self.validation_enabled = not self.validation_enabled
        status = "enabled" if self.validation_enabled else "disabled"
        print(f"\n📊 Comprehensive validation is now {status}")
        
        if not self.validation_enabled:
            print("⚠️ Note: Simple validation will be used instead")
            print("   Simple validation only checks if XML files exist and have content")
        else:
            print("✅ Comprehensive validation will check:")
            print("   • XML structure and format")
            print("   • Class names and values") 
            print("   • Bounding box coordinates")
            print("   • Negative values and coordinate bounds")
            print("   • Geometric validation")
            print("   • File consistency")
        
        return self.validation_enabled
        
    def load_configuration(self):
        """Load the batch labeling configuration."""
        print("=== Loading Batch Labeling Configuration ===")
        
        if not os.path.exists(self.config_file):
            print(f"❌ Configuration file not found: {self.config_file}")
            print("Please run Step2_organize_dataset.py first.")
            return False
        
        with open(self.config_file, 'r') as f:
            self.config_data = json.load(f)
        
        print(f"✅ Loaded configuration for {self.config_data['total_classes']} classes")
        print(f"✅ Total images to label: {self.config_data['total_images']}")
        
        return True
    
    def load_or_create_progress(self):
        """Load existing progress or create new progress tracking."""
        print("\n=== Loading Progress Tracking ===")
        
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress_data = json.load(f)
            print(f"✅ Loaded existing progress data")
        else:
            print("📝 Creating new progress tracking")
            self.progress_data = {
                'project_start_date': '2025-09-28',
                'last_updated': '',
                'total_classes': self.config_data['total_classes'],
                'total_images': self.config_data['total_images'],
                'completed_classes': 0,
                'completed_images': 0,
                'mandatory_progress': {},
                'cautionary_progress': {},
                'labeling_sessions': []
            }
        
        return True
    
    def validate_all_progress(self):
        """Validate and update progress for all classes to ensure accuracy."""
        print("\n=== Validating Progress Data ===")
        
        progress_updated = False
        
        # Validate mandatory classes
        for class_info in self.config_data.get('mandatory_road_signs', {}).get('classes', []):
            sequential_id = class_info['sequential_id']
            if sequential_id in self.progress_data.get('mandatory_progress', {}):
                current_progress = self.progress_data['mandatory_progress'][sequential_id]
                
                # Run validation
                valid_pairs, missing_xml, orphaned_xml = self.validate_image_annotation_pairs(class_info)
                
                # Check if progress data needs updating
                if (current_progress.get('labeled_images') != valid_pairs or
                    current_progress.get('missing_annotations') != len(missing_xml) or
                    current_progress.get('status') != ('completed' if valid_pairs == class_info['image_count'] else 'incomplete')):
                    
                    # Update progress data
                    self.progress_data['mandatory_progress'][sequential_id].update({
                        'labeled_images': valid_pairs,
                        'missing_annotations': len(missing_xml),
                        'orphaned_annotations': len(orphaned_xml),
                        'status': 'completed' if valid_pairs == class_info['image_count'] else 'incomplete',
                    })
                    progress_updated = True
                    print(f"🔧 Updated progress for {sequential_id}_{class_info['class_name']}: {valid_pairs}/{class_info['image_count']} labeled")
        
        # Validate cautionary classes
        for class_info in self.config_data.get('cautionary_road_signs', {}).get('classes', []):
            sequential_id = class_info['sequential_id']
            if sequential_id in self.progress_data.get('cautionary_progress', {}):
                current_progress = self.progress_data['cautionary_progress'][sequential_id]
                
                # Run validation
                valid_pairs, missing_xml, orphaned_xml = self.validate_image_annotation_pairs(class_info)
                
                # Check if progress data needs updating
                if (current_progress.get('labeled_images') != valid_pairs or
                    current_progress.get('missing_annotations') != len(missing_xml) or
                    current_progress.get('status') != ('completed' if valid_pairs == class_info['image_count'] else 'incomplete')):
                    
                    # Update progress data
                    self.progress_data['cautionary_progress'][sequential_id].update({
                        'labeled_images': valid_pairs,
                        'missing_annotations': len(missing_xml),
                        'orphaned_annotations': len(orphaned_xml),
                        'status': 'completed' if valid_pairs == class_info['image_count'] else 'incomplete',
                    })
                    progress_updated = True
                    print(f"🔧 Updated progress for {sequential_id}_{class_info['class_name']}: {valid_pairs}/{class_info['image_count']} labeled")
        
        # Update overall counts if any changes were made
        if progress_updated:
            self.progress_data['completed_images'] = sum(
                v.get('labeled_images', 0) 
                for progress in [self.progress_data.get('mandatory_progress', {}), self.progress_data.get('cautionary_progress', {})]
                for v in progress.values()
            )
            
            self.progress_data['completed_classes'] = len([
                v for progress in [self.progress_data.get('mandatory_progress', {}), self.progress_data.get('cautionary_progress', {})]
                for v in progress.values()
                if v.get('status') == 'completed'
            ])
            
            self.progress_data['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Save updated progress
            self.save_progress()
            print(f"✅ Progress validation completed - {self.progress_data['completed_images']}/{self.config_data['total_images']} images labeled")
        else:
            print("✅ Progress data is already accurate")
        
        return True
    
    def check_labelimg_installation(self):
        """Check if labelImg is properly installed and accessible."""
        print("\n=== Checking labelImg Installation ===")
        
        # First check if labelImg is available system-wide (Homebrew installation)
        try:
            result = subprocess.run(['which', 'labelImg'], capture_output=True, text=True)
            if result.returncode == 0:
                system_labelimg = result.stdout.strip()
                print(f"✅ System-wide labelImg found at: {system_labelimg}")
                self.labelimg_path = 'labelImg'  # Use system command
                return True
        except Exception:
            pass
        
        # Check if virtual environment exists
        if not os.path.exists(self.venv_python):
            print("❌ Virtual environment not found")
            print("💡 Please install labelImg system-wide or create virtual environment:")
            print("   brew install labelimg  (recommended for macOS)")
            print("   or")
            print("   cd /Users/jvarghese/Documents/TrafficSignProject")
            print("   python3 -m venv venv")
            print("   source venv/bin/activate")
            print("   pip install labelImg")
            return False
        
        # Check if labelImg is installed in the virtual environment
        if os.path.exists(self.labelimg_path):
            print(f"✅ labelImg found at: {self.labelimg_path}")
            return True
            
        # Try to check if it's installed as a Python package in venv
        try:
            subprocess.run([self.venv_python, '-c', 'import labelImg'], check=True, capture_output=True)
            print("✅ labelImg Python package is installed in virtual environment")
            self.labelimg_path = f'{self.venv_python} -m labelImg'
            return True
        except subprocess.CalledProcessError:
            pass
        
        print("❌ labelImg not found in virtual environment")
        print("\n🔧 Installing labelImg in virtual environment...")
        
        try:
            # Install labelImg in the virtual environment
            subprocess.run([self.venv_pip, 'install', 'labelImg'], check=True)
            print("✅ labelImg installed successfully in virtual environment")
            
            # Check if executable was created
            if os.path.exists(self.labelimg_path):
                return True
            else:
                self.labelimg_path = f'{self.venv_python} -m labelImg'
                return True
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install labelImg: {e}")
            print("💡 Try installing manually:")
            print("   brew install labelimg  (recommended for macOS)")
            print("   or")
            print("   source /Users/jvarghese/Documents/TrafficSignProject/venv/bin/activate")
            print("   pip install labelImg")
            return False
    
    def display_labeling_menu(self):
        """Display the main labeling menu with options."""
        print("\n" + "="*60)
        print("           🏷️  BATCH LABELING SYSTEM  🏷️")
        print("="*60)
        
        # Calculate progress statistics
        mandatory_completed = len([k for k, v in self.progress_data.get('mandatory_progress', {}).items() if v.get('status') == 'completed'])
        cautionary_completed = len([k for k, v in self.progress_data.get('cautionary_progress', {}).items() if v.get('status') == 'completed'])
        
        mandatory_total = self.config_data['mandatory_road_signs']['count']
        cautionary_total = self.config_data['cautionary_road_signs']['count']
        
        print(f"📊 Progress Overview:")
        print(f"   Mandatory Signs: {mandatory_completed}/{mandatory_total} classes completed ({self.config_data['mandatory_road_signs']['sequential_range']})")
        print(f"   Cautionary Signs: {cautionary_completed}/{cautionary_total} classes completed ({self.config_data['cautionary_road_signs']['sequential_range']})")
        print(f"   Total Images Labeled: {self.progress_data.get('completed_images', 0)}/{self.config_data['total_images']}")
        
        print(f"\n📋 Labeling Options:")
        print(f"   1. Label Mandatory Road Signs ({self.config_data['mandatory_road_signs']['sequential_range']})")
        print(f"   2. Label Cautionary Road Signs ({self.config_data['cautionary_road_signs']['sequential_range']})")
        print(f"   3. Label Specific Class (by sequential ID)")
        print(f"   4. Resume from Last Position")
        print(f"   5. View Detailed Progress")
        print(f"   6. Export Progress Report")
        print(f"   7. Exit")
        
        while True:
            try:
                choice = input(f"\n🎯 Choose an option (1-7): ").strip()
                if choice in ['1', '2', '3', '4', '5', '6', '7']:
                    return choice
                else:
                    print("❌ Invalid option. Please choose 1-7.")
            except (EOFError, KeyboardInterrupt):
                print("\n\n🛑 Labeling system interrupted by user")
                return '7'  # Exit gracefully
    
    def label_category(self, category):
        """Label all classes in a specific category."""
        if category == 'mandatory':
            classes_data = self.config_data['mandatory_road_signs']['classes']
            category_name = "Mandatory Road Signs"
        else:
            classes_data = self.config_data['cautionary_road_signs']['classes']
            category_name = "Cautionary Road Signs"
        
        print(f"\n🎯 Starting {category_name} Labeling")
        print(f"📝 Classes to label: {len(classes_data)}")
        
        for i, class_info in enumerate(classes_data, 1):
            sequential_id = class_info['sequential_id']
            original_id = class_info['original_id']
            class_name = class_info['class_name']

            print(f"\n📋 Class {i}/{len(classes_data)}: {sequential_id}_{class_name} (original: {original_id})")

            # Check if already completed (only skip if fully completed)
            progress_key = f"{category}_progress"
            current_status = None
            action_text = "Label"
            if (sequential_id in self.progress_data.get(progress_key, {})):
                current_status = self.progress_data[progress_key][sequential_id].get('status')
                if current_status == 'completed':
                    print(f"   ✅ Already completed - skipping")
                    # Run validation for this class
                    validation_results = self.validate_class_annotations(class_info)
                    if validation_results['total_errors'] == 0:
                        print(f"   🟢 Validation passed for {class_name}")
                    else:
                        print(f"   ⚠️ Validation issues detected for {class_name} ({validation_results['total_errors']} errors, {validation_results['total_warnings']} warnings)")
                    continue
                elif current_status == 'incomplete':
                    class_data = self.progress_data[progress_key][sequential_id]
                    labeled = class_data.get('labeled_images', 0)
                    missing = class_data.get('missing_annotations', 0)
                    print(f"   🔶 Previously started: {labeled}/{class_info['image_count']} images labeled, {missing} missing")
                    action_text = "Continue labeling"

            response = input(f"   🎯 {action_text} {class_name} ({class_info['image_count']} images)? [y/n/q]: ").strip().lower()

            if response == 'q':
                print("🛑 Stopping labeling process")
                break
            elif response == 'y':
                success = self.launch_labelimg_for_class(class_info)
                if success:
                    self.update_class_progress(category, sequential_id, class_info)
            else:
                print("   ⏭️  Skipped")
        
        print(f"\n🎉 {category_name} labeling session completed!")
    
    def label_specific_class(self):
        """Label a specific class chosen by the user."""
        print(f"\n🎯 Label Specific Class")
        
        # Show available classes with sequential IDs
        print(f"\n📋 Available Classes:")
        print(f"\n🔴 Mandatory Road Signs ({self.config_data['mandatory_road_signs']['sequential_range']}):")
        for class_info in self.config_data['mandatory_road_signs']['classes']:
            status = self.get_class_status('mandatory', class_info['sequential_id'])
            print(f"   {class_info['sequential_id']}: {class_info['class_name']} (orig: {class_info['original_id']}, {class_info['image_count']} images) - {status}")
        
        print(f"\n🔶 Cautionary Road Signs ({self.config_data['cautionary_road_signs']['sequential_range']}):")
        for class_info in self.config_data['cautionary_road_signs']['classes']:
            status = self.get_class_status('cautionary', class_info['sequential_id'])
            print(f"   {class_info['sequential_id']}: {class_info['class_name']} (orig: {class_info['original_id']}, {class_info['image_count']} images) - {status}")
        
        sequential_id = input(f"\n🎯 Enter Sequential Class ID to label (e.g., 001, 101): ").strip()
        
        # Find the class by sequential ID
        target_class = None
        category = None
        
        for class_info in self.config_data['mandatory_road_signs']['classes']:
            if class_info['sequential_id'] == sequential_id:
                target_class = class_info
                category = 'mandatory'
                break
        
        if not target_class:
            for class_info in self.config_data['cautionary_road_signs']['classes']:
                if class_info['sequential_id'] == sequential_id:
                    target_class = class_info
                    category = 'cautionary'
                    break
        
        if not target_class:
            print(f"❌ Sequential Class ID {sequential_id} not found")
            return
        
        print(f"\n📋 Selected: {target_class['sequential_id']}_{target_class['class_name']} (original ID: {target_class['original_id']})")
        print(f"📊 Images to label: {target_class['image_count']}")
        
        response = input(f"🎯 Start labeling? [y/n]: ").strip().lower()
        if response == 'y':
            success = self.launch_labelimg_for_class(target_class)
            if success:
                self.update_class_progress(category, sequential_id, target_class)
    
    def get_class_status(self, category, sequential_id):
        """Get the status of a specific class by sequential ID."""
        progress_key = f"{category}_progress"
        if sequential_id in self.progress_data.get(progress_key, {}):
            status = self.progress_data[progress_key][sequential_id].get('status', 'not_started')
            class_data = self.progress_data[progress_key][sequential_id]
            
            if status == 'completed':
                return "✅ Completed"
            elif status == 'in_progress':
                return "🔄 In Progress"
            elif status == 'incomplete':
                labeled = class_data.get('labeled_images', 0)
                total = class_data.get('total_images', 0)
                missing = class_data.get('missing_annotations', 0)
                return f"🔶 Incomplete ({labeled}/{total}, {missing} missing)"
        return "⭕ Not Started"
    
    def launch_labelimg_for_class(self, class_info):
        """Launch labelImg for a specific class."""
        images_dir = class_info['images_dir']
        annotations_dir = class_info['annotations_dir']
        classes_file = class_info['classes_file']
        
        # Create a symbolic link in the images directory pointing to the annotations directory
        # This ensures labelImg saves annotations in the correct location
        print(f"\n🚀 Launching labelImg for {class_info['class_name']}")
        print(f"📁 Images: {images_dir}")
        print(f"💾 Annotations: {annotations_dir}")
        print(f"🏷️  Classes: {classes_file}")
        
        # Ensure the annotations directory exists
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Create a predefined classes file with just the class name
        with open(classes_file, 'w') as f:
            f.write(f"{class_info['class_name']}")
        
        print(f"\n📝 Labeling Instructions:")
        print(f"   1. Draw bounding box around entire traffic sign")
        print(f"   2. Use class name: {class_info['class_name']}")
        print(f"   3. Save each annotation (Ctrl+S)")
        print(f"   4. Use A/D keys to navigate images")
        print(f"   5. ⚠️ IMPORTANT: When labelImg opens, IMMEDIATELY go to:")
        print(f"      File > Change Save Dir")
        print(f"      and set it to: {annotations_dir}")
        print(f"   6. Close labelImg when finished with all {class_info['image_count']} images")
            
        input(f"\n⏳ Press Enter when ready to launch labelImg...")
        
        try:
            # Launch labelImg with proper parameters
            cmd = [
                self.labelimg_path,
                images_dir,                    # Images directory
                classes_file,                  # Predefined classes
                annotations_dir                # Save directory (this should set it automatically)
            ]
            
            print(f"🎬 Starting labelImg...")
            print(f"    Images: {images_dir}")
            print(f"    Classes: {classes_file}")
            print(f"    Save Dir: {annotations_dir}")
            print(f"    📝 labelImg should automatically:")
            print(f"       - Load existing annotations")
            print(f"       - Use 'Truck_Prohibited' as default label")
            print(f"       - Save to correct annotations directory")
            result = subprocess.run(cmd, capture_output=False)
            
            print(f"\n🎯 labelImg session completed")
            
            # Check how many annotations were created in the annotations folder
            annotation_count = len([f for f in os.listdir(annotations_dir) if f.endswith('.xml')])
            
            # Also check if annotations were saved in the parent directory (common issue)
            class_dir = os.path.dirname(annotations_dir)
            parent_annotations = [f for f in os.listdir(class_dir) if f.endswith('.xml') and not os.path.isdir(os.path.join(class_dir, f))]
            
            if parent_annotations:
                print(f"� Moving {len(parent_annotations)} annotations from class folder to annotations folder...")
                for xml_file in parent_annotations:
                    src = os.path.join(class_dir, xml_file)
                    dst = os.path.join(annotations_dir, xml_file)
                    os.rename(src, dst)
                
                # Recount after moving
                annotation_count = len([f for f in os.listdir(annotations_dir) if f.endswith('.xml')])
            
            print(f"📊 Annotations found: {annotation_count}/{class_info['image_count']}")
            
            if annotation_count > 0:
                return True
            else:
                print(f"⚠️  No annotations found. Please make sure to save annotations.")
                return False
                
        except Exception as e:
            print(f"❌ Error launching labelImg: {e}")
            return False
    
    def validate_image_annotation_pairs(self, class_info):
        """Validate that each image has a corresponding XML annotation file with comprehensive checks."""
        if not self.validation_enabled:
            # Fall back to simple validation if comprehensive validation is disabled
            return self.simple_validate_image_annotation_pairs(class_info)
        
        # Run comprehensive validation
        validation_results = self.validate_class_annotations(class_info, show_details=False)
        
        valid_pairs = validation_results['valid']
        missing_xml = validation_results['missing_files']
        orphaned_xml = validation_results['orphaned_files']
        
        # If there are validation errors, show them and ask user what to do
        if validation_results['invalid'] > 0 or validation_results['total_errors'] > 0:
            print(f"\n⚠️ Validation issues found for {class_info['sequential_id']}_{class_info['class_name']}:")
            print(f"   Valid annotations: {validation_results['valid']}")
            print(f"   Invalid annotations: {validation_results['invalid']}")
            print(f"   Missing annotations: {validation_results['missing']}")
            print(f"   Total errors: {validation_results['total_errors']}")
            print(f"   Total warnings: {validation_results['total_warnings']}")
            
            if validation_results['invalid'] > 0:
                print(f"\n❌ Found {validation_results['invalid']} annotations with errors:")
                for detail in validation_results['error_details']:
                    print(f"   📄 {detail['file']}:")
                    for error in detail['errors']:
                        print(f"     ❌ {error}")
                
                print(f"\nWhat would you like to do?")
                print(f"1. Show detailed validation report")
                print(f"2. Attempt automatic fixes")
                print(f"3. Continue with current annotations (errors will cause training issues)")
                print(f"4. Re-label this class")
                
                while True:
                    choice = input("Enter your choice (1-4): ").strip()
                    if choice == '1':
                        self.validate_class_annotations(class_info, show_details=True)
                        break
                    elif choice == '2':
                        if self.fix_annotation_issues(class_info, validation_results):
                            # Re-validate after fixes
                            validation_results = self.validate_class_annotations(class_info, show_details=False)
                            valid_pairs = validation_results['valid']
                            missing_xml = validation_results['missing_files']
                            orphaned_xml = validation_results['orphaned_files']
                        break
                    elif choice == '3':
                        print("⚠️ Continuing with current annotations. This may cause training issues.")
                        break
                    elif choice == '4':
                        print("🔄 Please re-label this class to fix the annotation issues.")
                        return 0, missing_xml, orphaned_xml  # Return 0 valid pairs to indicate need for re-labeling
                    else:
                        print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        
        return valid_pairs, missing_xml, orphaned_xml
    
    def simple_validate_image_annotation_pairs(self, class_info):
        """Simple validation (original implementation) - fallback method."""
        images_dir = class_info['images_dir']
        annotations_dir = class_info['annotations_dir']
        
        if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
            return 0, [], []
        
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Get all XML files
        xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
        
        # Check for missing XML files
        missing_xml = []
        valid_pairs = 0
        
        for img_file in image_files:
            img_base = os.path.splitext(img_file)[0]
            xml_file = img_base + '.xml'
            
            if xml_file in xml_files:
                # Simply check if XML file exists and has some content
                xml_path = os.path.join(annotations_dir, xml_file)
                try:
                    if os.path.getsize(xml_path) > 0:
                        valid_pairs += 1
                    else:
                        missing_xml.append(f"{img_file} (empty XML)")
                except Exception:
                    missing_xml.append(f"{img_file} (corrupted XML)")
            else:
                missing_xml.append(f"{img_file} (no XML)")
        
        # Check for orphaned XML files
        orphaned_xml = []
        image_bases = {os.path.splitext(img)[0] for img in image_files}
        for xml_file in xml_files:
            xml_base = os.path.splitext(xml_file)[0]
            if xml_base not in image_bases:
                orphaned_xml.append(xml_file)
        
        return valid_pairs, missing_xml, orphaned_xml

    def update_class_progress(self, category, sequential_id, class_info):
        """Update progress for a completed class using sequential ID with proper validation."""
        progress_key = f"{category}_progress"
        
        if progress_key not in self.progress_data:
            self.progress_data[progress_key] = {}
        
        # Use proper validation instead of just counting XML files
        valid_pairs, missing_xml, orphaned_xml = self.validate_image_annotation_pairs(class_info)
        
        # Show comprehensive validation report
        print(f"")  # Add spacing
        self.validate_class_annotations(class_info, show_details=True)
        
        self.progress_data[progress_key][sequential_id] = {
            'sequential_id': sequential_id,
            'original_id': class_info.get('original_id', 'unknown'),
            'class_name': class_info['class_name'],
            'total_images': class_info['image_count'],
            'labeled_images': valid_pairs,  # Use valid pairs instead of XML count
            'missing_annotations': len(missing_xml),
            'orphaned_annotations': len(orphaned_xml),
            'status': 'completed' if valid_pairs == class_info['image_count'] else 'incomplete',
            'completion_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'annotations_dir': class_info['annotations_dir']
        }
        
        # Update overall progress
        self.progress_data['completed_images'] = sum(
            v.get('labeled_images', 0) 
            for progress in [self.progress_data.get('mandatory_progress', {}), self.progress_data.get('cautionary_progress', {})]
            for v in progress.values()
        )
        
        self.progress_data['completed_classes'] = len([
            v for progress in [self.progress_data.get('mandatory_progress', {}), self.progress_data.get('cautionary_progress', {})]
            for v in progress.values()
            if v.get('status') == 'completed'
        ])
        
        self.progress_data['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save progress
        self.save_progress()
        
        print(f"✅ Progress updated for {sequential_id}_{class_info['class_name']} (original: {class_info.get('original_id', 'unknown')})")
        print(f"📊 Total progress: {self.progress_data['completed_images']}/{self.config_data['total_images']} images labeled")
    
    def save_progress(self):
        """Save current progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress_data, f, indent=2)
    
    def view_detailed_progress(self):
        """Display detailed progress report."""
        print(f"\n" + "="*60)
        print(f"           📊 DETAILED PROGRESS REPORT 📊")
        print(f"="*60)
        
        print(f"\n🔴 Mandatory Road Signs Progress ({self.config_data['mandatory_road_signs']['sequential_range']}):")
        mandatory_progress = self.progress_data.get('mandatory_progress', {})
        for class_info in self.config_data['mandatory_road_signs']['classes']:
            sequential_id = class_info['sequential_id']
            original_id = class_info['original_id']
            if sequential_id in mandatory_progress:
                progress = mandatory_progress[sequential_id]
                status_icon = "✅" if progress['status'] == 'completed' else "⚠️" if progress['status'] == 'incomplete' else "🔄"
                missing_info = f" (Missing: {progress.get('missing_annotations', 0)})" if progress.get('missing_annotations', 0) > 0 else ""
                print(f"   {status_icon} {sequential_id}: {progress['class_name']} (orig: {original_id}) - {progress['labeled_images']}/{progress['total_images']}{missing_info}")
            else:
                print(f"   ⭕ {sequential_id}: {class_info['class_name']} (orig: {original_id}) - Not Started")
        
        print(f"\n🔶 Cautionary Road Signs Progress ({self.config_data['cautionary_road_signs']['sequential_range']}):")
        cautionary_progress = self.progress_data.get('cautionary_progress', {})
        for class_info in self.config_data['cautionary_road_signs']['classes']:
            sequential_id = class_info['sequential_id']
            original_id = class_info['original_id']
            if sequential_id in cautionary_progress:
                progress = cautionary_progress[sequential_id]
                status_icon = "✅" if progress['status'] == 'completed' else "⚠️" if progress['status'] == 'incomplete' else "🔄"
                missing_info = f" (Missing: {progress.get('missing_annotations', 0)})" if progress.get('missing_annotations', 0) > 0 else ""
                print(f"   {status_icon} {sequential_id}: {progress['class_name']} (orig: {original_id}) - {progress['labeled_images']}/{progress['total_images']}{missing_info}")
            else:
                print(f"   ⭕ {sequential_id}: {class_info['class_name']} (orig: {original_id}) - Not Started")
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total Classes: {self.progress_data['completed_classes']}/{self.config_data['total_classes']} completed")
        print(f"   Total Images: {self.progress_data['completed_images']}/{self.config_data['total_images']} labeled")
        
        completion_rate = (self.progress_data['completed_images'] / self.config_data['total_images']) * 100
        print(f"   Completion Rate: {completion_rate:.1f}%")
        
        if self.progress_data.get('last_updated'):
            print(f"   Last Updated: {self.progress_data['last_updated']}")
    
    def export_progress_report(self):
        """Export detailed progress report to file."""
        report_file = os.path.join(self.organized_dataset_dir, 'labeling_progress_report.md')
        
        report_content = f"""# Traffic Sign Labeling Progress Report

## Project Overview
- **Project:** {self.config_data['project_name']}
- **Report Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Last Updated:** {self.progress_data.get('last_updated', 'Never')}

## Overall Progress
- **Total Classes:** {self.progress_data['completed_classes']}/{self.config_data['total_classes']} ({(self.progress_data['completed_classes']/self.config_data['total_classes']*100):.1f}%)
- **Total Images:** {self.progress_data['completed_images']}/{self.config_data['total_images']} ({(self.progress_data['completed_images']/self.config_data['total_images']*100):.1f}%)

## Mandatory Road Signs Progress
"""
        
        mandatory_progress = self.progress_data.get('mandatory_progress', {})
        for class_info in self.config_data['mandatory_road_signs']['classes']:
            sequential_id = class_info['sequential_id']
            original_id = class_info['original_id']
            if sequential_id in mandatory_progress:
                progress = mandatory_progress[sequential_id]
                status = "✅ Completed" if progress['status'] == 'completed' else "🔄 In Progress"
                report_content += f"\n- **{sequential_id}**: {progress['class_name']} (orig: {original_id}) - {progress['labeled_images']}/{progress['total_images']} images - {status}"
            else:
                report_content += f"\n- **{sequential_id}**: {class_info['class_name']} (orig: {original_id}) - Not Started"
        
        report_content += f"\n\n## Cautionary Road Signs Progress\n"
        
        cautionary_progress = self.progress_data.get('cautionary_progress', {})
        for class_info in self.config_data['cautionary_road_signs']['classes']:
            sequential_id = class_info['sequential_id']
            original_id = class_info['original_id']
            if sequential_id in cautionary_progress:
                progress = cautionary_progress[sequential_id]
                status = "✅ Completed" if progress['status'] == 'completed' else "🔄 In Progress"
                report_content += f"\n- **{sequential_id}**: {progress['class_name']} (orig: {original_id}) - {progress['labeled_images']}/{progress['total_images']} images - {status}"
            else:
                report_content += f"\n- **{sequential_id}**: {class_info['class_name']} (orig: {original_id}) - Not Started"
        
        report_content += f"""

## Next Steps
- Complete labeling for remaining classes
- Run Step 4: Image Augmentation
- Run Step 5: YOLO Format Conversion

---
*Generated by Step3_batch_labeling.py*
"""
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"✅ Progress report exported to: {report_file}")


def main():
    config_file = '/Users/jvarghese/Documents/TrafficSignProject/organized_dataset/batch_labeling_config.json'

    print("=== Step 3: Batch Labeling System ===")

    manager = BatchLabelingManager(config_file)

    # Load configuration
    if not manager.load_configuration():
        return

    # Load or create progress tracking
    if not manager.load_or_create_progress():
        return

    # Validate and update progress data to ensure accuracy
    if not manager.validate_all_progress():
        return

    # Check labelImg installation
    if not manager.check_labelimg_installation():
        print("❌ Cannot proceed without labelImg")
        return

    # Main labeling loop
    while True:
        try:
            choice = manager.display_labeling_menu()
            
            if choice == '1':
                manager.label_category('mandatory')
            elif choice == '2':
                manager.label_category('cautionary')
            elif choice == '3':
                manager.label_specific_class()
            elif choice == '4':
                print("🔄 Resume functionality - choose category or specific class")
            elif choice == '5':
                manager.view_detailed_progress()
            elif choice == '6':
                manager.export_progress_report()
            elif choice == '7':
                print("👋 Exiting Batch Labeling System")
                break
            else:
                print("❌ Invalid option. Please choose 1-7.")
            
            if choice in ['1', '2', '3']:
                input("\n⏳ Press Enter to return to main menu...")
                
        except KeyboardInterrupt:
            print("\n\n🛑 Labeling system interrupted by user")
            print("💾 Progress has been saved")
            break
    
    print(f"\n🎯 Next Step: Once labeling is complete, run Step4_image_augmentation.py")


if __name__ == "__main__":
    main()