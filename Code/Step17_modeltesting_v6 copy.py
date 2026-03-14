#!/usr/bin/env python3.9
"""
Step 14: Model Testing Script - v5
Comprehensive test evaluation with metrics, plots, and visualizations
Compatible with M3 Pro MacBook configuration
For Step10 YOLOv12n Advanced Training v5 (Progressive Knowledge Distillation)
"""

import os
import random
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from ultralytics import YOLO

# === CONFIGURATION ===
DATASET_BASE = "/Users/jvarghese/Documents/TrafficSignProject/yolo_dataset"
# Step8 v3 uses staged training - check for best model in stage3, stage2, or stage1
TRAIN_BASE_DIR = "/Users/jvarghese/Documents/TrafficSignProject/output/yolo_output_v6/runs/train/Step4_traffic_signs_yolo12s_yolo12n/"
TEST_DIR = "/Users/jvarghese/Documents/TrafficSignProject/output/yolo_output_v6/runs/test_imagesize_1024/traffic_signs_yolo12n_v6"
DEVICE = "mps"  # GPU mode for M3 Pro

# Evaluation Parameters
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 1024    
BATCH_SIZE = 16
WORKERS = 4
NUM_SAMPLE_IMAGES = 6  # For visualization grid


def find_best_model():
    """
    Find the best model from Step8 v3 staged training by comparing metrics from CSV files.
    Compares mAP@0.5 from all stages and selects the one with highest performance.
    """
    stages = ["stage1", "stage2", "stage3"]
    stage_metrics = []
    
    print("🔍 Analyzing all stages to find best model...")
    print()
    
    for stage_name in stages:
        stage_dir = os.path.join(TRAIN_BASE_DIR, stage_name)
        results_csv = os.path.join(stage_dir, "results.csv")
        best_pt = os.path.join(stage_dir, "weights", "best.pt")
        
        # Check if both CSV and model exist
        if not os.path.exists(results_csv) or not os.path.exists(best_pt):
            print(f"   ⚠️  {stage_name}: Missing results.csv or best.pt, skipping...")
            continue
        
        try:
            # Read results CSV
            df = pd.read_csv(results_csv)
            
            if len(df) == 0:
                print(f"   ⚠️  {stage_name}: Empty results.csv, skipping...")
                continue
            
            # Find mAP@0.5 column (try different possible column names)
            map50_col = None
            for col in ['metrics/mAP50(B)', 'mAP50(B)', 'val/mAP50', 'mAP@0.5']:
                if col in df.columns:
                    map50_col = col
                    break
            
            if map50_col is None:
                print(f"   ⚠️  {stage_name}: No mAP@0.5 column found in results.csv, skipping...")
                continue
            
            # Get maximum mAP@0.5 value (best performance)
            map50_values = df[map50_col].dropna()
            if len(map50_values) == 0:
                print(f"   ⚠️  {stage_name}: No valid mAP@0.5 values, skipping...")
                continue
            
            max_map50 = map50_values.max()
            final_map50 = map50_values.iloc[-1]  # Last epoch value
            
            # Also get mAP@0.5:0.95 if available
            map50_95_col = None
            max_map50_95 = None
            for col in ['metrics/mAP50-95(B)', 'mAP50-95(B)', 'val/mAP50-95', 'mAP@0.5:0.95']:
                if col in df.columns:
                    map50_95_col = col
                    map50_95_values = df[col].dropna()
                    if len(map50_95_values) > 0:
                        max_map50_95 = map50_95_values.max()
                    break
            
            stage_metrics.append({
                'stage': stage_name,
                'path': best_pt,
                'max_map50': max_map50,
                'final_map50': final_map50,
                'max_map50_95': max_map50_95,
                'epochs': len(df)
            })
            
            print(f"   ✅ {stage_name}: Max mAP@0.5 = {max_map50:.4f}, Final mAP@0.5 = {final_map50:.4f}, Epochs = {len(df)}")
            
        except Exception as e:
            print(f"   ❌ {stage_name}: Error reading results.csv - {e}")
            continue
    
    print()
    
    if len(stage_metrics) == 0:
        print("❌ No valid stages found with results.csv and best.pt files")
        return None, None
    
    # Sort by max_map50 (descending) to find best performing stage
    stage_metrics.sort(key=lambda x: x['max_map50'], reverse=True)
    best_stage = stage_metrics[0]
    
    print("=" * 60)
    print("📊 STAGE COMPARISON RESULTS")
    print("=" * 60)
    for i, stage in enumerate(stage_metrics):
        marker = "🏆" if i == 0 else "  "
        print(f"{marker} {stage['stage']}: Max mAP@0.5 = {stage['max_map50']:.4f}, Final = {stage['final_map50']:.4f}")
    print("=" * 60)
    print()
    
    print(f"✅ Selected best model: {best_stage['stage']}")
    print(f"   Max mAP@0.5: {best_stage['max_map50']:.4f}")
    print(f"   Final mAP@0.5: {best_stage['final_map50']:.4f}")
    if best_stage['max_map50_95'] is not None:
        print(f"   Max mAP@0.5:0.95: {best_stage['max_map50_95']:.4f}")
    print(f"   Model path: {best_stage['path']}")
    print()
    
    return best_stage['path'], best_stage['stage']


def run_comprehensive_test_evaluation():
    """Run comprehensive test evaluation with metrics, plots, and visualizations"""
    
    print("🔍 TEST PREDICTIONS: Testing Best Model on Test Dataset")
    print("=" * 60)
    
    # Find the best model from staged training
    MODEL_PATH, stage_name = find_best_model()
    
    if MODEL_PATH is None:
        print(f"❌ Best model not found in any stage at: {TRAIN_BASE_DIR}")
        print("   Please check:")
        print(f"      - {TRAIN_BASE_DIR}/stage3/weights/best.pt")
        print(f"      - {TRAIN_BASE_DIR}/stage2/weights/best.pt")
        print(f"      - {TRAIN_BASE_DIR}/stage1/weights/best.pt")
        print("   Please run training first")
        return False
    
    print(f"✅ Found best model from {stage_name}: {MODEL_PATH}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Best model not found at: {MODEL_PATH}")
        print("   Please run training first")
        return False
    
    # Create test output directory
    os.makedirs(TEST_DIR, exist_ok=True)
    
    print(f"✅ Loading best model: {MODEL_PATH}")
    print(f"✅ Test results will be saved to: {TEST_DIR}\n")
    
    # Load the best model
    try:
        best_model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Dataset YAML path
    dataset_yaml = os.path.join(DATASET_BASE, "dataset.yaml")
    
    if not os.path.exists(dataset_yaml):
        print(f"❌ Dataset YAML not found: {dataset_yaml}")
        return False
    
    print("🚀 Running predictions on test dataset...")
    print(f"   Dataset: {dataset_yaml}")
    print(f"   Device: {DEVICE}")
    print(f"   Image Size: {IMAGE_SIZE}")
    print(f"   Output: {TEST_DIR}")
    print()
    
    # Run test predictions with comprehensive evaluation
    try:
        test_results = best_model.val(
            data=dataset_yaml,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=WORKERS,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            plots=True,  # Generate ALL test plots
            save_json=True,  # Save results as JSON (COCO format)
            save_txt=True,  # Save prediction labels
            save_conf=True,  # Save confidence scores in labels
            save_crop=False,  # Don't save cropped predictions (too many files)
            project=os.path.dirname(TEST_DIR),
            name=os.path.basename(TEST_DIR),
            exist_ok=True,
            verbose=True,
            rect=False,  # Evaluate at native resolution
            split='test'  # Use test split
        )
        
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS")
        print("=" * 60)
        
        # Display key metrics
        print(f"\n🎯 Overall Performance:")
        print(f"   mAP@0.5:      {test_results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        print(f"   mAP@0.5:0.95: {test_results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"   Precision:    {test_results.results_dict.get('metrics/precision(B)', 0):.4f}")
        print(f"   Recall:       {test_results.results_dict.get('metrics/recall(B)', 0):.4f}")
        
        # Display per-class metrics if available
        if hasattr(test_results, 'ap_class_index') and test_results.ap_class_index is not None:
            print(f"\n📋 Per-Class Performance:")
            
            # Get class names from dataset.yaml
            with open(dataset_yaml, 'r') as f:
                dataset_config = yaml.safe_load(f)
                class_names = dataset_config.get('names', [])
            
            # Display top 10 and bottom 10 classes by mAP
            if hasattr(test_results, 'ap50') and test_results.ap50 is not None:
                ap50_values = test_results.ap50
                
                # Sort classes by mAP@0.5
                class_performance = []
                for idx, ap in enumerate(ap50_values):
                    class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
                    class_performance.append((class_name, ap))
                
                class_performance.sort(key=lambda x: x[1], reverse=True)
                
                print(f"\n   Top 10 Classes (by mAP@0.5):")
                for i, (class_name, ap) in enumerate(class_performance[:10]):
                    print(f"      {i+1}. {class_name}: {ap:.4f}")
                
                if len(class_performance) > 10:
                    print(f"\n   Bottom 10 Classes (by mAP@0.5):")
                    for i, (class_name, ap) in enumerate(class_performance[-10:]):
                        print(f"      {len(class_performance)-9+i}. {class_name}: {ap:.4f}")
        
        print("\n" + "=" * 60)
        print(f"✅ Test predictions completed successfully!")
        print(f"📁 Test results saved to: {TEST_DIR}/")
        print("=" * 60)
        
        # List all generated plots and outputs
        print(f"\n📊 Generated Test Outputs:")
        print(f"\n   🎯 Performance Curves:")
        print(f"      📈 Confusion Matrix: confusion_matrix.png")
        print(f"      📈 Confusion Matrix (normalized): confusion_matrix_normalized.png")
        print(f"      📈 Precision-Recall Curve: PR_curve.png")
        print(f"      📈 F1 Score Curve: F1_curve.png")
        print(f"      📈 Precision Curve: P_curve.png")
        print(f"      📈 Recall Curve: R_curve.png")
        print(f"\n   📊 Classification Metrics:")
        print(f"      📋 Results CSV: results.csv")
        print(f"      📋 JSON Results (COCO format): predictions.json")
        print(f"      📄 Prediction Labels: labels/ (with confidence scores)")
        print(f"\n   🖼️  Visual Outputs:")
        print(f"      🖼️  Test Batch Images: test_batch*_pred.jpg")
        print(f"      🖼️  Test Batch Labels: test_batch*_labels.jpg")
        
        # Generate additional classification analysis plots
        print(f"\n📊 Generating additional classification analysis...")
        
        # Get class names and metrics
        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
            class_names = dataset_config.get('names', [])
        
        # Create per-class performance analysis
        if hasattr(test_results, 'box') and hasattr(test_results.box, 'all_ap'):
            # Get per-class metrics
            ap50 = test_results.box.all_ap[:, 0] if hasattr(test_results.box, 'all_ap') else []
            ap75 = test_results.box.all_ap[:, 5] if hasattr(test_results.box, 'all_ap') and test_results.box.all_ap.shape[1] > 5 else []
            
            if len(ap50) > 0:
                # Plot 1: Per-class mAP@0.5 bar chart
                fig, ax = plt.subplots(figsize=(16, 10))
                
                # Sort classes by mAP
                sorted_indices = ap50.argsort()[::-1]
                sorted_ap50 = ap50[sorted_indices]
                sorted_names = [class_names[i] if i < len(class_names) else f"Class {i}" for i in sorted_indices]
                
                colors = ['green' if ap > 0.7 else 'orange' if ap > 0.5 else 'red' for ap in sorted_ap50]
                
                bars = ax.barh(range(len(sorted_ap50)), sorted_ap50, color=colors, alpha=0.7)
                ax.set_yticks(range(len(sorted_ap50)))
                ax.set_yticklabels(sorted_names, fontsize=8)
                ax.set_xlabel('mAP@0.5', fontsize=12, fontweight='bold')
                ax.set_title('Per-Class Performance (mAP@0.5) - Test Dataset - Sorted by Performance', fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                ax.set_xlim([0, 1])
                
                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, sorted_ap50)):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center', fontsize=7)
                
                plt.tight_layout()
                class_perf_path = os.path.join(TEST_DIR, "per_class_performance.png")
                plt.savefig(class_perf_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   ✅ Per-class performance chart: per_class_performance.png")
                
                # Plot 2: Performance distribution histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(ap50, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                ax.axvline(ap50.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ap50.mean():.3f}')
                ax.axvline(np.median(ap50), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(ap50):.3f}')
                ax.set_xlabel('mAP@0.5', fontsize=12, fontweight='bold')
                ax.set_ylabel('Number of Classes', fontsize=12, fontweight='bold')
                ax.set_title('Distribution of Per-Class Performance (mAP@0.5) - Test Dataset', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                plt.tight_layout()
                dist_path = os.path.join(TEST_DIR, "performance_distribution.png")
                plt.savefig(dist_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   ✅ Performance distribution: performance_distribution.png")
                
                # Plot 3: Top 20 and Bottom 20 classes comparison
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                
                # Top 20
                top_20_indices = sorted_indices[:20]
                top_20_ap = ap50[top_20_indices]
                top_20_names = [class_names[i] if i < len(class_names) else f"Class {i}" for i in top_20_indices]
                
                ax1.barh(range(len(top_20_ap)), top_20_ap, color='green', alpha=0.7)
                ax1.set_yticks(range(len(top_20_ap)))
                ax1.set_yticklabels(top_20_names, fontsize=10)
                ax1.set_xlabel('mAP@0.5', fontsize=12, fontweight='bold')
                ax1.set_title('Top 20 Best Performing Classes - Test Dataset', fontsize=14, fontweight='bold', color='green')
                ax1.grid(axis='x', alpha=0.3)
                ax1.set_xlim([0, 1])
                
                # Bottom 20
                bottom_20_indices = sorted_indices[-20:][::-1]
                bottom_20_ap = ap50[bottom_20_indices]
                bottom_20_names = [class_names[i] if i < len(class_names) else f"Class {i}" for i in bottom_20_indices]
                
                ax2.barh(range(len(bottom_20_ap)), bottom_20_ap, color='red', alpha=0.7)
                ax2.set_yticks(range(len(bottom_20_ap)))
                ax2.set_yticklabels(bottom_20_names, fontsize=10)
                ax2.set_xlabel('mAP@0.5', fontsize=12, fontweight='bold')
                ax2.set_title('Bottom 20 Worst Performing Classes - Test Dataset', fontsize=14, fontweight='bold', color='red')
                ax2.grid(axis='x', alpha=0.3)
                ax2.set_xlim([0, 1])
                
                plt.tight_layout()
                comparison_path = os.path.join(TEST_DIR, "top_bottom_classes_comparison.png")
                plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   ✅ Top/bottom classes comparison: top_bottom_classes_comparison.png")
                
                # Save detailed per-class metrics to CSV
                metrics_df = pd.DataFrame({
                    'Class': [class_names[i] if i < len(class_names) else f"Class {i}" for i in range(len(ap50))],
                    'mAP@0.5': ap50,
                    'mAP@0.75': ap75 if len(ap75) > 0 else [0] * len(ap50)
                })
                metrics_df = metrics_df.sort_values('mAP@0.5', ascending=False)
                metrics_csv_path = os.path.join(TEST_DIR, "per_class_metrics.csv")
                metrics_df.to_csv(metrics_csv_path, index=False)
                print(f"   ✅ Per-class metrics CSV: per_class_metrics.csv")
                
                print(f"\n   📊 Performance Statistics:")
                print(f"      Mean mAP@0.5: {ap50.mean():.4f}")
                print(f"      Median mAP@0.5: {np.median(ap50):.4f}")
                print(f"      Std Dev: {ap50.std():.4f}")
                print(f"      Min: {ap50.min():.4f}")
                print(f"      Max: {ap50.max():.4f}")
                print(f"      Classes > 0.7: {(ap50 > 0.7).sum()} / {len(ap50)}")
                print(f"      Classes > 0.5: {(ap50 > 0.5).sum()} / {len(ap50)}")
                print(f"      Classes < 0.3: {(ap50 < 0.3).sum()} / {len(ap50)}")
        else:
            print(f"   ⚠️  Per-class metrics not available in test results")
        
        # Generate sample predictions on test images
        print(f"\n🖼️  Generating sample prediction visualizations...")
        
        # Get test images
        test_images_dir = os.path.join(DATASET_BASE, "images", "test")
        if os.path.exists(test_images_dir):
            # Get list of test images
            test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(test_images) > 0:
                # Select random images for visualization
                num_samples = min(NUM_SAMPLE_IMAGES, len(test_images))
                sample_images = random.sample(test_images, num_samples)
                
                print(f"   Visualizing predictions on {num_samples} sample images...")
                
                # Create prediction output directory
                pred_vis_dir = os.path.join(TEST_DIR, "sample_predictions")
                os.makedirs(pred_vis_dir, exist_ok=True)
                
                # Run predictions and save visualizations
                for idx, img_name in enumerate(sample_images):
                    img_path = os.path.join(test_images_dir, img_name)
                    
                    # Run prediction
                    results = best_model.predict(
                        source=img_path,
                        conf=CONFIDENCE_THRESHOLD,
                        iou=IOU_THRESHOLD,
                        device=DEVICE,
                        save=True,
                        save_conf=True,
                        project=pred_vis_dir,
                        name=f"sample_{idx+1}",
                        exist_ok=True,
                        show_labels=True,
                        show_conf=True,
                        line_width=2,
                        verbose=False
                    )
                
                print(f"   ✅ Sample predictions saved to: {pred_vis_dir}/")
                print(f"   📁 View individual predictions: {pred_vis_dir}/sample_*/")
                
                # Create a summary visualization
                print(f"\n📊 Creating prediction summary grid...")
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle('Sample Test Predictions - YOLO12n v3 Model (Progressive Knowledge Distillation)', fontsize=16, fontweight='bold')
                
                for idx in range(num_samples):
                    row = idx // 3
                    col = idx % 3
                    ax = axes[row, col]
                    
                    # Find the predicted image
                    pred_img_path = os.path.join(pred_vis_dir, f"sample_{idx+1}", sample_images[idx])
                    
                    if os.path.exists(pred_img_path):
                        img = Image.open(pred_img_path)
                        ax.imshow(img)
                        ax.set_title(f"Sample {idx+1}: {sample_images[idx][:20]}...", fontsize=10)
                        ax.axis('off')
                    else:
                        ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
                        ax.axis('off')
                
                plt.tight_layout()
                summary_path = os.path.join(TEST_DIR, "prediction_summary.png")
                plt.savefig(summary_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Prediction summary saved to: {summary_path}")
            else:
                print(f"   ⚠️  No test images found in {test_images_dir}")
        else:
            print(f"   ⚠️  Test images directory not found: {test_images_dir}")
        
        print("\n" + "=" * 60)
        print(f"📊 All test outputs complete!")
        print(f"📁 Main directory: {TEST_DIR}/")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test predictions failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"   Please check the dataset and model paths")
        return False


if __name__ == "__main__":
    print("🔍 YOLO MODEL TEST PREDICTIONS - v3")
    print("=" * 60)
    
    # Run comprehensive test evaluation
    success = run_comprehensive_test_evaluation()
    
    if success:
        print("\n✅ Test predictions completed successfully!")
    else:
        print("\n❌ Test predictions failed!")

