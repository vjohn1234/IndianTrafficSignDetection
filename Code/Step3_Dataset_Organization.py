#!/usr/bin/env python3
"""
Step 2: Dataset Organization
Organizes images by traffic sign classes based on Step 1 analysis.
Creates structured workspace for efficient batch labeling.
"""

import os
import shutil
import json
from pathlib import Path


class DatasetOrganizer:
    def __init__(self, source_dataset_path, analysis_dir):
        self.source_dataset_path = source_dataset_path
        self.analysis_dir = analysis_dir
        self.output_dir = '/Users/jvarghese/Documents/TrafficSignProject/organized_dataset'
        self.classes_data = {}
        
    def load_analysis_results(self):
        """Load results from Step 1 analysis."""
        print("=== Loading Step 1 Analysis Results ===")
        
        classes_file = os.path.join(self.analysis_dir, 'classes_for_labeling.json')
        if not os.path.exists(classes_file):
            print(f"❌ Analysis file not found: {classes_file}")
            print("Please run Step1_data_analysis.py first.")
            return False
        
        with open(classes_file, 'r') as f:
            self.classes_data = json.load(f)
        
        print(f"✅ Loaded data for {len(self.classes_data)} classes with images")
        return True
    
    def create_organized_structure(self):
        """Create organized directory structure with sequential numbering."""
        print(f"\n=== Creating Organized Dataset Structure with Sequential Numbering ===")
        print(f"Output directory: {self.output_dir}")
        
        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create category directories
        mandatory_dir = os.path.join(self.output_dir, "Mandatory_Road_Signs")
        cautionary_dir = os.path.join(self.output_dir, "Cautionary_Road_Signs")
        os.makedirs(mandatory_dir, exist_ok=True)
        os.makedirs(cautionary_dir, exist_ok=True)
        
        # Separate and sort classes by category
        mandatory_classes = [(k, v) for k, v in self.classes_data.items() if v['category'] == 'Mandatory']
        cautionary_classes = [(k, v) for k, v in self.classes_data.items() if v['category'] == 'Cautionary']
        
        mandatory_classes.sort(key=lambda x: int(x[0]))
        cautionary_classes.sort(key=lambda x: int(x[0]))
        
        created_dirs = 0
        self.sequential_mapping = {}  # Store original ID to sequential ID mapping
        
        # Create Mandatory Road Signs with sequential numbering (001, 002, 003...)
        print(f"\n📂 Creating Mandatory Road Signs directories:")
        for i, (original_id, class_info) in enumerate(mandatory_classes, 1):
            sequential_id = f"{i:03d}"  # Format as 001, 002, 003...
            class_name = class_info['class_name']
            
            # Create class directory with sequential naming
            class_dir_name = f"{sequential_id}_{class_name}"
            class_dir_path = os.path.join(mandatory_dir, class_dir_name)
            
            # Store mapping for later reference
            self.sequential_mapping[original_id] = {
                'sequential_id': sequential_id,
                'category': 'Mandatory',
                'class_info': class_info
            }
            
            # Create subdirectories for the labeling workflow
            self._create_class_subdirectories(class_dir_path)
            
            # Create class-specific files with sequential ID
            self.create_class_files(class_dir_path, class_name, class_info, sequential_id, original_id)
            
            print(f"  ✅ {sequential_id}_{class_name} (original: {original_id}, {class_info['image_count']} images)")
            created_dirs += 1
        
        # Create Cautionary Road Signs with sequential numbering (101, 102, 103...)
        print(f"\n📂 Creating Cautionary Road Signs directories:")
        for i, (original_id, class_info) in enumerate(cautionary_classes, 1):
            sequential_id = f"{100 + i:03d}"  # Format as 101, 102, 103...
            class_name = class_info['class_name']
            
            # Create class directory with sequential naming
            class_dir_name = f"{sequential_id}_{class_name}"
            class_dir_path = os.path.join(cautionary_dir, class_dir_name)
            
            # Store mapping for later reference
            self.sequential_mapping[original_id] = {
                'sequential_id': sequential_id,
                'category': 'Cautionary',
                'class_info': class_info
            }
            
            # Create subdirectories for the labeling workflow
            self._create_class_subdirectories(class_dir_path)
            
            # Create class-specific files with sequential ID
            self.create_class_files(class_dir_path, class_name, class_info, sequential_id, original_id)
            
            print(f"  ✅ {sequential_id}_{class_name} (original: {original_id}, {class_info['image_count']} images)")
            created_dirs += 1
        
        print(f"\n📁 Total directories created: {created_dirs}")
        print(f"📂 Mandatory Road Signs: {len(mandatory_classes)} classes (001-{len(mandatory_classes):03d})")
        print(f"📂 Cautionary Road Signs: {len(cautionary_classes)} classes (101-{100 + len(cautionary_classes):03d})")
        
        # Save the sequential mapping for reference
        self.save_sequential_mapping()
        
        return created_dirs
    
    def _create_class_subdirectories(self, class_dir_path):
        """Create subdirectories for the labeling workflow."""
        subdirs = [
            'images',                    # Original images
            'annotations',              # XML annotations from labelImg
            'augmented_images',         # Generated augmented images
            'augmented_annotations',    # Annotations for augmented images
            'yolo_labels'              # YOLO format labels
        ]
        
        for subdir in subdirs:
            subdir_path = os.path.join(class_dir_path, subdir)
            os.makedirs(subdir_path, exist_ok=True)
    
    def create_class_files(self, class_dir_path, class_name, class_info, sequential_id=None, original_id=None):
        """Create configuration files for each class."""
        
        # Create classes.txt file for labelImg
        classes_file = os.path.join(class_dir_path, 'classes.txt')
        with open(classes_file, 'w') as f:
            f.write(class_name)
        
        # Create class info file with both sequential and original IDs
        class_info_file = os.path.join(class_dir_path, 'class_info.json')
        info_data = {
            'sequential_id': sequential_id or 'unknown',
            'original_id': original_id or 'unknown', 
            'class_name': class_name,
            'category': class_info['category'],
            'image_count': class_info['image_count'],
            'target_total_images': 500,
            'augmentation_needed': max(0, 500 - class_info['image_count']),
            'labeling_status': 'not_started',
            'augmentation_status': 'not_started'
        }
        
        with open(class_info_file, 'w') as f:
            json.dump(info_data, f, indent=2)
        
        # Create labeling guide with ID information
        self.create_labeling_guide(class_dir_path, class_name, class_info, sequential_id, original_id)
    
    def create_labeling_guide(self, class_dir_path, class_name, class_info, sequential_id=None, original_id=None):
        """Create a labeling guide for each class."""
        guide_content = f"""# Labeling Guide: {class_name}

## Class Information
- **Sequential ID:** {sequential_id or 'N/A'} (folder number)
- **Original ID:** {original_id or 'N/A'} (dataset class ID)
- **Class Name:** {class_name}
- **Category:** {class_info['category']} Road Signs
- **Images to Label:** {class_info['image_count']}
- **Target Total Images:** 500 (including augmented)

## Labeling Instructions
1. **Open labelImg** and configure:
   - Open Dir: `{os.path.join(class_dir_path, 'images')}`
   - Change Save Dir: `{os.path.join(class_dir_path, 'annotations')}`
   - Enable Auto Save mode

2. **For each image:**
   - Draw bounding box around the entire traffic sign
   - Use class name: `{class_name}`
   - Include sign border but minimize background
   - Be consistent with box placement

3. **Quality Guidelines:**
   - ✅ Tight bounding box around complete sign
   - ✅ Include any text/symbols on the sign
   - ✅ Consistent labeling across all images
   - ❌ Don't include too much background
   - ❌ Don't cut off parts of the sign

## Keyboard Shortcuts
- **W**: Create bounding box
- **A**: Previous image
- **D**: Next image
- **Ctrl+S**: Save annotation
- **Delete**: Remove selected box

## Progress Tracking
- Total images: {class_info['image_count']}
- Completed: 0/{class_info['image_count']}
- Status: Not started

---
*Generated by Step2_organize_dataset.py*
"""
        
        guide_file = os.path.join(class_dir_path, 'labeling_guide.md')
        with open(guide_file, 'w') as f:
            f.write(guide_content)
    
    def copy_images_to_classes(self):
        """Copy images from source dataset to organized class directories using sequential mapping."""
        print(f"\n=== Copying and Renaming Images to Sequential Class Directories ===")
        
        total_copied = 0
        
        # Process Mandatory Road Signs with sequential mapping
        print(f"\n📂 Processing Mandatory Road Signs:")
        mandatory_classes = [(k, v) for k, v in self.classes_data.items() if v['category'] == 'Mandatory']
        mandatory_classes.sort(key=lambda x: int(x[0]))
        
        for i, (original_id, class_info) in enumerate(mandatory_classes, 1):
            sequential_id = f"{i:03d}"
            class_name = class_info['class_name']
            images = class_info['images']
            
            # Determine destination directory using sequential ID
            class_dir_name = f"{sequential_id}_{class_name}"
            images_dir = os.path.join(self.output_dir, "Mandatory_Road_Signs", class_dir_name, 'images')
            
            copied_count = self._copy_class_images(images, images_dir, sequential_id)
            print(f"  ✅ {sequential_id} ({class_name}) [orig: {original_id}]: {copied_count}/{len(images)} images copied and renamed")
            total_copied += copied_count
        
        # Process Cautionary Road Signs with sequential mapping
        print(f"\n📂 Processing Cautionary Road Signs:")
        cautionary_classes = [(k, v) for k, v in self.classes_data.items() if v['category'] == 'Cautionary']
        cautionary_classes.sort(key=lambda x: int(x[0]))
        
        for i, (original_id, class_info) in enumerate(cautionary_classes, 1):
            sequential_id = f"{100 + i:03d}"
            class_name = class_info['class_name']
            images = class_info['images']
            
            # Determine destination directory using sequential ID
            class_dir_name = f"{sequential_id}_{class_name}"
            images_dir = os.path.join(self.output_dir, "Cautionary_Road_Signs", class_dir_name, 'images')
            
            copied_count = self._copy_class_images(images, images_dir, sequential_id)
            print(f"  ✅ {sequential_id} ({class_name}) [orig: {original_id}]: {copied_count}/{len(images)} images copied and renamed")
            total_copied += copied_count
        
        print(f"\n📊 Total images copied and renamed: {total_copied}")
        print(f"🎯 All image filenames now match their sequential folder structure")
        return total_copied
    
    def _copy_class_images(self, images, destination_dir, sequential_id):
        """Copy and rename images for a specific class to match sequential ID."""
        copied_count = 0
        for i, image_file in enumerate(images, 1):
            source_path = os.path.join(self.source_dataset_path, image_file)
            
            # Create new filename with sequential ID instead of original ID
            # Original: 013_001.jpg -> New: 007_001.jpg (if sequential_id is 007)
            original_extension = os.path.splitext(image_file)[1]  # .jpg
            new_filename = f"{sequential_id}_{i:03d}{original_extension}"  # 007_001.jpg
            dest_path = os.path.join(destination_dir, new_filename)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            else:
                print(f"    ⚠️  Image not found: {source_path}")
        
        return copied_count
    
    def save_sequential_mapping(self):
        """Save the sequential ID mapping for reference."""
        mapping_file = os.path.join(self.output_dir, 'sequential_id_mapping.json')
        
        mapping_data = {
            'description': 'Mapping between original dataset class IDs and sequential folder IDs',
            'mandatory_mapping': {},
            'cautionary_mapping': {},
            'creation_date': '2025-09-28'
        }
        
        for original_id, mapping_info in self.sequential_mapping.items():
            if mapping_info['category'] == 'Mandatory':
                mapping_data['mandatory_mapping'][original_id] = {
                    'sequential_id': mapping_info['sequential_id'],
                    'class_name': mapping_info['class_info']['class_name'],
                    'image_count': mapping_info['class_info']['image_count']
                }
            else:
                mapping_data['cautionary_mapping'][original_id] = {
                    'sequential_id': mapping_info['sequential_id'],
                    'class_name': mapping_info['class_info']['class_name'],
                    'image_count': mapping_info['class_info']['image_count']
                }
        
        with open(mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f"✅ Sequential ID mapping saved: {mapping_file}")
    
    def create_batch_labeling_config(self):
        """Create configuration for batch labeling workflow with sequential IDs."""
        print(f"\n=== Creating Batch Labeling Configuration ===")
        
        # Create lists using sequential mapping
        mandatory_list = []
        cautionary_list = []
        
        # Process Mandatory Road Signs with sequential numbering
        mandatory_classes = [(k, v) for k, v in self.classes_data.items() if v['category'] == 'Mandatory']
        mandatory_classes.sort(key=lambda x: int(x[0]))
        
        for i, (original_id, class_info) in enumerate(mandatory_classes, 1):
            sequential_id = f"{i:03d}"
            class_name = class_info['class_name']
            class_dir_name = f"{sequential_id}_{class_name}"
            category_dir = "Mandatory_Road_Signs"
            
            mandatory_list.append({
                'priority': i,
                'sequential_id': sequential_id,
                'original_id': original_id,
                'class_name': class_name,
                'category': 'Mandatory Road Signs',
                'image_count': class_info['image_count'],
                'class_directory': os.path.join(self.output_dir, category_dir, class_dir_name),
                'images_dir': os.path.join(self.output_dir, category_dir, class_dir_name, 'images'),
                'annotations_dir': os.path.join(self.output_dir, category_dir, class_dir_name, 'annotations'),
                'classes_file': os.path.join(self.output_dir, category_dir, class_dir_name, 'classes.txt')
            })
        
        # Process Cautionary Road Signs with sequential numbering
        cautionary_classes = [(k, v) for k, v in self.classes_data.items() if v['category'] == 'Cautionary']
        cautionary_classes.sort(key=lambda x: int(x[0]))
        
        for i, (original_id, class_info) in enumerate(cautionary_classes, 1):
            sequential_id = f"{100 + i:03d}"
            class_name = class_info['class_name']
            class_dir_name = f"{sequential_id}_{class_name}"
            category_dir = "Cautionary_Road_Signs"
            
            cautionary_list.append({
                'priority': 100 + i,
                'sequential_id': sequential_id,
                'original_id': original_id,
                'class_name': class_name,
                'category': 'Cautionary Road Signs',
                'image_count': class_info['image_count'],
                'class_directory': os.path.join(self.output_dir, category_dir, class_dir_name),
                'images_dir': os.path.join(self.output_dir, category_dir, class_dir_name, 'images'),
                'annotations_dir': os.path.join(self.output_dir, category_dir, class_dir_name, 'annotations'),
                'classes_file': os.path.join(self.output_dir, category_dir, class_dir_name, 'classes.txt')
            })
        
        # Save labeling configuration
        config_data = {
            'project_name': 'TrafficSignProject',
            'total_classes': len(self.classes_data),
            'total_images': sum(info['image_count'] for info in self.classes_data.values()),
            'organized_dataset_dir': self.output_dir,
            'sequential_numbering': True,
            'mandatory_road_signs': {
                'count': len(mandatory_list),
                'sequential_range': f"001-{len(mandatory_list):03d}",
                'classes': mandatory_list
            },
            'cautionary_road_signs': {
                'count': len(cautionary_list),
                'sequential_range': f"101-{100 + len(cautionary_list):03d}",
                'classes': cautionary_list
            },
            'target_images_per_class': 500,
            'created_date': '2025-09-28'
        }
        
        config_file = os.path.join(self.output_dir, 'batch_labeling_config.json')
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Batch labeling configuration saved: {config_file}")
        print(f"📂 Mandatory Road Signs: {len(mandatory_list)} classes ({config_data['mandatory_road_signs']['sequential_range']})")
        print(f"📂 Cautionary Road Signs: {len(cautionary_list)} classes ({config_data['cautionary_road_signs']['sequential_range']})")
        
        # Create quick reference summary
        self.create_organization_summary()
        
        return config_file
    
    def create_organization_summary(self):
        """Create a summary report of the sequential organization."""
        summary_content = f"""# Traffic Sign Dataset Organization Summary (Sequential Numbering)

## Overview
- **Total Classes:** {len(self.classes_data)}
- **Total Images:** {sum(info['image_count'] for info in self.classes_data.values())}
- **Target per Class:** 500 images (original + augmented)
- **Organization Date:** 2025-09-28
- **Sequential Numbering:** ✅ Enabled (gaps removed)

## Dataset Structure
```
organized_dataset/
├── Mandatory_Road_Signs/
│   ├── 001_Stop/                   # Original ID: 001
│   ├── 002_Give_Way/               # Original ID: 002
│   ├── 003_No_Entry/               # Original ID: 003
│   └── ... (continues sequentially)
├── Cautionary_Road_Signs/
│   ├── 101_Right_Hand_Curve/       # Original ID: 101
│   ├── 102_Left_Hand_Curve/        # Original ID: 102
│   ├── 103_Right_Hair_Pin_Bend/    # Original ID: 103
│   └── ... (continues sequentially)
├── sequential_id_mapping.json     # Maps original to sequential IDs
└── batch_labeling_config.json     # Configuration for Step 3
```

## Sequential Mapping Summary

### Mandatory Road Signs (001-{len([k for k, v in self.classes_data.items() if v['category'] == 'Mandatory']):03d})
**Sequential → Original ID mapping:**"""
        
        # Add Mandatory signs with sequential mapping
        mandatory_classes = [(k, v) for k, v in self.classes_data.items() if v['category'] == 'Mandatory']
        mandatory_classes.sort(key=lambda x: int(x[0]))
        
        for i, (original_id, class_info) in enumerate(mandatory_classes, 1):
            sequential_id = f"{i:03d}"
            summary_content += f"\n- **{sequential_id}**: {class_info['class_name']} (orig: {original_id}) - {class_info['image_count']} images"
        
        summary_content += f"""

### Cautionary Road Signs (101-{100 + len([k for k, v in self.classes_data.items() if v['category'] == 'Cautionary']):03d})
**Sequential → Original ID mapping:**"""
        
        # Add Cautionary signs with sequential mapping  
        cautionary_classes = [(k, v) for k, v in self.classes_data.items() if v['category'] == 'Cautionary']
        cautionary_classes.sort(key=lambda x: int(x[0]))
        
        for i, (original_id, class_info) in enumerate(cautionary_classes, 1):
            sequential_id = f"{100 + i:03d}"
            summary_content += f"\n- **{sequential_id}**: {class_info['class_name']} (orig: {original_id}) - {class_info['image_count']} images"
        
        mandatory_count = len(mandatory_classes)
        cautionary_count = len(cautionary_classes)
        mandatory_images = sum(v[1]['image_count'] for v in mandatory_classes)
        cautionary_images = sum(v[1]['image_count'] for v in cautionary_classes)
        
        summary_content += f"""

## Summary Statistics
- **Mandatory Road Signs:** {mandatory_count} classes, {mandatory_images} images (001-{mandatory_count:03d})
- **Cautionary Road Signs:** {cautionary_count} classes, {cautionary_images} images (101-{100 + cautionary_count:03d})

## Key Benefits of Sequential Numbering
✅ **No gaps in folder sequence**  
✅ **Easier navigation and organization**  
✅ **Consistent numbering for labeling workflow**  
✅ **Original IDs preserved in mapping file**  

## Next Steps
1. **Run Step 3:** Batch labeling system (updated for sequential IDs)
2. **Label all images:** Use labelImg with organized structure  
3. **Run Step 4:** Generate augmented images (target: 500 per class)
4. **Run Step 5:** Convert to YOLO format

## Files Created
- `batch_labeling_config.json` - Configuration for Step 3 (with sequential IDs)
- `sequential_id_mapping.json` - Original to sequential ID mapping
- Each class directory has its own labeling guide and configuration

---
*Generated by Step2_organize_dataset.py (Sequential Version)*
"""
        
        summary_file = os.path.join(self.output_dir, 'organization_summary.md')
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        print(f"✅ Organization summary saved: {summary_file}")


def main():
    source_dataset_path = '/Users/jvarghese/Documents/TrafficSignProject/DataSet'
    analysis_dir = '/Users/jvarghese/Documents/TrafficSignProject/Code/analysis'
    
    print("=== Step 2: Dataset Organization ===")
    
    organizer = DatasetOrganizer(source_dataset_path, analysis_dir)
    
    # Load analysis results from Step 1
    if not organizer.load_analysis_results():
        return
    
    # Create organized structure
    created_dirs = organizer.create_organized_structure()
    
    if created_dirs == 0:
        print("❌ No directories created. Check the analysis results.")
        return
    
    # Copy images to class directories
    total_copied = organizer.copy_images_to_classes()
    
    # Create batch labeling configuration
    config_file = organizer.create_batch_labeling_config()
    
    print(f"\n🎉 Step 2 Organization Complete!")
    print(f"✅ Organized {len(organizer.classes_data)} classes")
    print(f"✅ Copied {total_copied} images")
    print(f"✅ Created batch labeling configuration")
    print(f"✅ Dataset ready for Step 3: Batch Labeling")
    
    print(f"\n📁 Organized dataset location:")
    print(f"   {organizer.output_dir}")
    
    print(f"\n🚀 Next: Run Step3_batch_labeling.py")


if __name__ == "__main__":
    main()