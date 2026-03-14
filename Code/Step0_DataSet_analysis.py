#!/usr/bin/env python3
"""
Step 1: Traffic Sign Dataset Analysis
Analyzes the complete dataset and creates class mappings based on the provided classification data.
Skips classes with zero images and creates descriptive class names for YOLO training.
"""

import os
import re
import json
from collections import defaultdict, Counter


class TrafficSignAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.output_dir = '/Users/jvarghese/Documents/TrafficSignProject/Code/analysis'
        
        # Traffic sign classification data from the provided tables
        self.sign_classifications = {
            # Mandatory Road Signs (001-041)
            '001': {'category': 'Mandatory', 'name': 'Stop', 'expected_count': 10},
            '002': {'category': 'Mandatory', 'name': 'Give_Way', 'expected_count': 10},
            '003': {'category': 'Mandatory', 'name': 'No_Entry', 'expected_count': 10},
            '004': {'category': 'Mandatory', 'name': 'Priority_for_Oncoming_Traffic', 'expected_count': 10},
            '005': {'category': 'Mandatory', 'name': 'All_Motor_Vehicles_Prohibited', 'expected_count': 10},
            '006': {'category': 'Mandatory', 'name': 'Truck_Prohibited', 'expected_count': 10},
            '007': {'category': 'Mandatory', 'name': 'Bullock_Hand_Cart_Prohibited', 'expected_count': 0},
            '008': {'category': 'Mandatory', 'name': 'Bullock_Cart_Prohibited', 'expected_count': 0},
            '009': {'category': 'Mandatory', 'name': 'Tongas_Prohibited', 'expected_count': 0},
            '010': {'category': 'Mandatory', 'name': 'Hand_Cart_Prohibited', 'expected_count': 0},
            '011': {'category': 'Mandatory', 'name': 'Cycle_Prohibited', 'expected_count': 0},
            '012': {'category': 'Mandatory', 'name': 'Pedestrians_Prohibited', 'expected_count': 0},
            '013': {'category': 'Mandatory', 'name': 'Right_Turn_Prohibited', 'expected_count': 10},
            '014': {'category': 'Mandatory', 'name': 'Left_Turn_Prohibited', 'expected_count': 10},
            '015': {'category': 'Mandatory', 'name': 'U_Turn_Prohibited', 'expected_count': 10},
            '016': {'category': 'Mandatory', 'name': 'Overtaking_Prohibited', 'expected_count': 10},
            '017': {'category': 'Mandatory', 'name': 'Horn_Prohibited', 'expected_count': 10},
            '018': {'category': 'Mandatory', 'name': 'Width_Limit', 'expected_count': 0},
            '019': {'category': 'Mandatory', 'name': 'Height_Limit', 'expected_count': 0},
            '020': {'category': 'Mandatory', 'name': 'Length_Limit', 'expected_count': 0},
            '021': {'category': 'Mandatory', 'name': 'Load_Limit', 'expected_count': 0},
            '022': {'category': 'Mandatory', 'name': 'Axle_Load_Limit', 'expected_count': 0},
            '023': {'category': 'Mandatory', 'name': 'Speed_Limit_20', 'expected_count': 10},
            '024': {'category': 'Mandatory', 'name': 'No_Parking', 'expected_count': 10},
            '025': {'category': 'Mandatory', 'name': 'No_Stopping_or_Standing', 'expected_count': 10},
            '026': {'category': 'Mandatory', 'name': 'Compulsory_Turn_Left', 'expected_count': 10},
            '027': {'category': 'Mandatory', 'name': 'Compulsory_Turn_Right', 'expected_count': 10},
            '028': {'category': 'Mandatory', 'name': 'Compulsory_Ahead_Only', 'expected_count': 0},
            '029': {'category': 'Mandatory', 'name': 'Compulsory_Turn_Right_Ahead', 'expected_count': 10},
            '030': {'category': 'Mandatory', 'name': 'Compulsory_Turn_Left_Ahead', 'expected_count': 10},
            '031': {'category': 'Mandatory', 'name': 'Compulsory_Ahead_or_Turn_Right', 'expected_count': 10},
            '032': {'category': 'Mandatory', 'name': 'Compulsory_Ahead_or_Turn_Left', 'expected_count': 10},
            '033': {'category': 'Mandatory', 'name': 'Compulsory_Keep_Left', 'expected_count': 10},
            '034': {'category': 'Mandatory', 'name': 'Compulsory_Cycle_Track', 'expected_count': 10},
            '035': {'category': 'Mandatory', 'name': 'Compulsory_Sound_Horn', 'expected_count': 10},
            '036': {'category': 'Mandatory', 'name': 'Compulsory_Minimum_Speed', 'expected_count': 0},
            '037': {'category': 'Mandatory', 'name': 'Restriction_Ends', 'expected_count': 0},
            '038': {'category': 'Mandatory', 'name': 'Mandatory_Two_Way_Traffic', 'expected_count': 10},
            '039': {'category': 'Mandatory', 'name': 'Speed_Limit_40', 'expected_count': 10},
            '040': {'category': 'Mandatory', 'name': 'Pass_Either_Side', 'expected_count': 10},
            '041': {'category': 'Mandatory', 'name': 'Speed_Limit_60', 'expected_count': 10},
            
            # Cautionary Road Signs (101-137)
            '101': {'category': 'Cautionary', 'name': 'Right_Hand_Curve', 'expected_count': 15},
            '102': {'category': 'Cautionary', 'name': 'Left_Hand_Curve', 'expected_count': 15},
            '103': {'category': 'Cautionary', 'name': 'Right_Hair_Pin_Bend', 'expected_count': 15},
            '104': {'category': 'Cautionary', 'name': 'Left_Hair_Pin_Bend', 'expected_count': 15},
            '105': {'category': 'Cautionary', 'name': 'Right_Reverse_Bend', 'expected_count': 10},
            '106': {'category': 'Cautionary', 'name': 'Left_Reverse_Bend', 'expected_count': 15},
            '107': {'category': 'Cautionary', 'name': 'Steep_Ascent', 'expected_count': 10},
            '108': {'category': 'Cautionary', 'name': 'Steep_Descent', 'expected_count': 10},
            '109': {'category': 'Cautionary', 'name': 'Narrow_Road_Ahead', 'expected_count': 10},
            '110': {'category': 'Cautionary', 'name': 'Road_Widens_Ahead', 'expected_count': 0},
            '111': {'category': 'Cautionary', 'name': 'Narrow_Bridge', 'expected_count': 10},
            '112': {'category': 'Cautionary', 'name': 'Slippery_Road', 'expected_count': 10},
            '113': {'category': 'Cautionary', 'name': 'Loose_Gravel', 'expected_count': 10},
            '114': {'category': 'Cautionary', 'name': 'Cycle_Crossing', 'expected_count': 10},
            '115': {'category': 'Cautionary', 'name': 'Pedestrian_Crossing', 'expected_count': 10},
            '116': {'category': 'Cautionary', 'name': 'School_Ahead', 'expected_count': 10},
            '117': {'category': 'Cautionary', 'name': 'Traffic_Signal', 'expected_count': 0},
            '118': {'category': 'Cautionary', 'name': 'Cattle', 'expected_count': 10},
            '119': {'category': 'Cautionary', 'name': 'Ferry', 'expected_count': 0},
            '120': {'category': 'Cautionary', 'name': 'Falling_Rocks', 'expected_count': 10},
            '121': {'category': 'Cautionary', 'name': 'Dangerous_Dip', 'expected_count': 10},
            '122': {'category': 'Cautionary', 'name': 'Hump_or_Rough_Road', 'expected_count': 10},
            '123': {'category': 'Cautionary', 'name': 'Barrier_Ahead', 'expected_count': 10},
            '124': {'category': 'Cautionary', 'name': 'Gap_in_Median', 'expected_count': 10},
            '125': {'category': 'Cautionary', 'name': 'Cross_Road', 'expected_count': 10},
            '126': {'category': 'Cautionary', 'name': 'Side_Road_Left', 'expected_count': 10},
            '127': {'category': 'Cautionary', 'name': 'Side_Road_Right', 'expected_count': 10},
            '128': {'category': 'Cautionary', 'name': 'Y_Intersection', 'expected_count': 10},
            '129': {'category': 'Cautionary', 'name': 'T_Intersection', 'expected_count': 10},
            '130': {'category': 'Cautionary', 'name': 'Staggered_Intersection', 'expected_count': 10},
            '131': {'category': 'Cautionary', 'name': 'Round_About', 'expected_count': 10},
            '132': {'category': 'Cautionary', 'name': 'Guarded_Level_Crossing', 'expected_count': 10},
            '133': {'category': 'Cautionary', 'name': 'Unguarded_Level_Crossing', 'expected_count': 10},
            '134': {'category': 'Cautionary', 'name': 'Quayside_or_River_Bank', 'expected_count': 0},
            '135': {'category': 'Cautionary', 'name': 'Men_at_Work', 'expected_count': 10},
            '136': {'category': 'Cautionary', 'name': 'U_Turn_Ahead', 'expected_count': 10},
            '137': {'category': 'Cautionary', 'name': 'Merge_Traffic_Ahead', 'expected_count': 10}
        }
        
        self.found_classes = {}
        self.classes_with_images = {}
        self.classes_without_images = {}
        
    def analyze_dataset(self):
        """Analyze the dataset and create class mappings."""
        print("=== Step 1: Traffic Sign Dataset Analysis ===")
        print(f"Dataset path: {self.dataset_path}")
        print()
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset path does not exist: {self.dataset_path}")
            return False
            
        # Get all image files
        image_files = [f for f in os.listdir(self.dataset_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"📊 Total image files found: {len(image_files)}")
        
        # Parse filenames and group by class
        class_pattern = re.compile(r'^(\d{3})_(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)
        
        for image_file in image_files:
            match = class_pattern.match(image_file)
            if match:
                class_id = match.group(1)
                if class_id not in self.found_classes:
                    self.found_classes[class_id] = []
                self.found_classes[class_id].append(image_file)
            else:
                print(f"⚠️  Unrecognized filename format: {image_file}")
        
        # Separate classes with and without images
        for class_id, class_info in self.sign_classifications.items():
            if class_id in self.found_classes:
                actual_count = len(self.found_classes[class_id])
                expected_count = class_info['expected_count']
                
                if actual_count > 0:
                    self.classes_with_images[class_id] = {
                        'category': class_info['category'],
                        'name': class_info['name'],
                        'expected_count': expected_count,
                        'actual_count': actual_count,
                        'images': self.found_classes[class_id],
                        'status': 'OK' if actual_count == expected_count else f'Expected {expected_count}, found {actual_count}'
                    }
            else:
                if class_info['expected_count'] == 0:
                    self.classes_without_images[class_id] = {
                        'category': class_info['category'],
                        'name': class_info['name'],
                        'reason': 'Expected 0 images'
                    }
                else:
                    self.classes_without_images[class_id] = {
                        'category': class_info['category'], 
                        'name': class_info['name'],
                        'reason': f'Expected {class_info["expected_count"]} images but none found'
                    }
        
        return True
    
    def generate_statistics(self):
        """Generate detailed statistics."""
        print("\n=== Dataset Statistics ===")
        
        total_classes_with_images = len(self.classes_with_images)
        total_classes_without_images = len(self.classes_without_images)
        total_images = sum(info['actual_count'] for info in self.classes_with_images.values())
        
        print(f"🏷️  Classes with images: {total_classes_with_images}")
        print(f"🚫 Classes without images: {total_classes_without_images}")
        print(f"📷 Total images: {total_images}")
        
        if total_classes_with_images > 0:
            avg_images = total_images / total_classes_with_images
            print(f"📈 Average images per class: {avg_images:.1f}")
        
        # Category breakdown
        mandatory_classes = sum(1 for info in self.classes_with_images.values() if info['category'] == 'Mandatory')
        cautionary_classes = sum(1 for info in self.classes_with_images.values() if info['category'] == 'Cautionary')
        
        mandatory_images = sum(info['actual_count'] for info in self.classes_with_images.values() if info['category'] == 'Mandatory')
        cautionary_images = sum(info['actual_count'] for info in self.classes_with_images.values() if info['category'] == 'Cautionary')
        
        print(f"\n📊 Category Breakdown:")
        print(f"   Mandatory Signs: {mandatory_classes} classes, {mandatory_images} images")
        print(f"   Cautionary Signs: {cautionary_classes} classes, {cautionary_images} images")
        
        # Show classes with images
        print(f"\n📋 Classes with Images (for labeling):")
        for class_id in sorted(self.classes_with_images.keys()):
            info = self.classes_with_images[class_id]
            print(f"   {class_id}: {info['name']} ({info['actual_count']} images) - {info['status']}")
        
        # Show classes without images (to be skipped)
        if self.classes_without_images:
            print(f"\n⏭️  Classes to Skip (no images):")
            for class_id in sorted(self.classes_without_images.keys()):
                info = self.classes_without_images[class_id]
                print(f"   {class_id}: {info['name']} - {info['reason']}")
    
    def calculate_augmentation_requirements(self):
        """Calculate how many augmented images needed per class to reach 500 total."""
        print(f"\n=== Augmentation Requirements (Target: 500 images per class) ===")
        
        augmentation_plan = {}
        
        for class_id, info in self.classes_with_images.items():
            original_count = info['actual_count']
            target_count = 500
            augmented_needed = max(0, target_count - original_count)
            augmentation_factor = augmented_needed / original_count if original_count > 0 else 0
            
            augmentation_plan[class_id] = {
                'class_name': info['name'],
                'original_images': original_count,
                'augmented_needed': augmented_needed,
                'total_target': target_count,
                'augmentation_factor': f"{augmentation_factor:.1f}x" if augmentation_factor > 0 else "None needed"
            }
            
            print(f"   {class_id} ({info['name']}): {original_count} → {target_count} images ({augmented_needed} augmented needed)")
        
        return augmentation_plan
    
    def save_analysis_results(self):
        """Save all analysis results to files."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save classes with images (for labeling and augmentation)
        labeling_classes = {}
        for class_id, info in self.classes_with_images.items():
            labeling_classes[class_id] = {
                'class_name': info['name'],
                'category': info['category'],
                'image_count': info['actual_count'],
                'images': info['images']
            }
        
        labeling_file = os.path.join(self.output_dir, 'classes_for_labeling.json')
        with open(labeling_file, 'w') as f:
            json.dump(labeling_classes, f, indent=2)
        
        # Save YOLO class mapping
        yolo_classes = []
        for class_id in sorted(self.classes_with_images.keys()):
            yolo_classes.append(self.classes_with_images[class_id]['name'])
        
        yolo_classes_file = os.path.join(self.output_dir, 'yolo_classes.txt')
        with open(yolo_classes_file, 'w') as f:
            for class_name in yolo_classes:
                f.write(f"{class_name}\n")
        
        # Save class ID to name mapping
        class_mapping = {}
        for class_id, info in self.classes_with_images.items():
            class_mapping[class_id] = info['name']
        
        mapping_file = os.path.join(self.output_dir, 'class_id_to_name_mapping.json')
        with open(mapping_file, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        # Save augmentation requirements
        augmentation_plan = self.calculate_augmentation_requirements()
        augmentation_file = os.path.join(self.output_dir, 'augmentation_requirements.json')
        with open(augmentation_file, 'w') as f:
            json.dump(augmentation_plan, f, indent=2)
        
        # Save complete analysis summary
        summary = {
            'analysis_date': '2025-09-28',
            'total_classes_with_images': len(self.classes_with_images),
            'total_classes_without_images': len(self.classes_without_images),
            'total_images': sum(info['actual_count'] for info in self.classes_with_images.values()),
            'mandatory_classes': len([c for c in self.classes_with_images.values() if c['category'] == 'Mandatory']),
            'cautionary_classes': len([c for c in self.classes_with_images.values() if c['category'] == 'Cautionary']),
            'target_images_per_class': 500,
            'pipeline_ready': True
        }
        
        summary_file = os.path.join(self.output_dir, 'analysis_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Analysis results saved to: {self.output_dir}")
        print(f"   📄 classes_for_labeling.json - Classes that need labeling")
        print(f"   📄 yolo_classes.txt - YOLO class names")
        print(f"   📄 class_id_to_name_mapping.json - ID to name mapping")
        print(f"   📄 augmentation_requirements.json - Augmentation plan")
        print(f"   📄 analysis_summary.json - Complete analysis summary")
        
        return True


def main():
    dataset_path = '/Users/jvarghese/Documents/TrafficSignProject/DataSet'
    
    analyzer = TrafficSignAnalyzer(dataset_path)
    
    if analyzer.analyze_dataset():
        analyzer.generate_statistics()
        analyzer.save_analysis_results()
        
        print(f"\n🎉 Step 1 Analysis Complete!")
        print(f"✅ Found {len(analyzer.classes_with_images)} classes with images")
        print(f"✅ Ready for Step 2: Dataset Organization")
        
    else:
        print(f"❌ Analysis failed. Please check the dataset path.")


if __name__ == "__main__":
    main()