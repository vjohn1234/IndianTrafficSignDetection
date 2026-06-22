#!/usr/bin/env python3
"""
Generate info.labels JSON for yolo_dataset/images/test from yolo_dataset/labels/test.
YOLO format: class_id x_center y_center width height (normalized 0-1).
Output: path, name, category, label, boundingBoxes (x, y, width, height in pixels).
"""

import json
import re
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "yolo_dataset" / "images" / "test"
LABELS_DIR = PROJECT_ROOT / "yolo_dataset" / "labels" / "test"
DATASET_YAML = PROJECT_ROOT / "yolo_dataset" / "dataset.yaml"
OUTPUT_PATH = PROJECT_ROOT / "yolo_dataset" / "images" / "test" / "info.labels"

# Class names from dataset.yaml (index = class_id)
CLASS_NAMES = [
    "Stop", "Give_Way", "No_Entry", "Priority_for_Oncoming_Traffic", "All_Motor_Vehicles_Prohibited",
    "Truck_Prohibited", "Right_Turn_Prohibited", "Left_Turn_Prohibited", "U_Turn_Prohibited", "Overtaking_Prohibited",
    "Horn_Prohibited", "Speed_Limit_20", "No_Parking", "No_Stopping_or_Standing", "Compulsory_Turn_Left",
    "Compulsory_Turn_Right", "Compulsory_Turn_Right_Ahead", "Compulsory_Turn_Left_Ahead", "Compulsory_Ahead_or_Turn_Right", "Compulsory_Ahead_or_Turn_Left",
    "Compulsory_Keep_Left", "Compulsory_Cycle_Track", "Compulsory_Sound_Horn", "Mandatory_Two_Way_Traffic", "Speed_Limit_40",
    "Pass_Either_Side", "Speed_Limit_60", "Right_Hand_Curve", "Left_Hand_Curve", "Right_Hair_Pin_Bend",
    "Left_Hair_Pin_Bend", "Right_Reverse_Bend", "Left_Reverse_Bend", "Steep_Ascent", "Steep_Descent",
    "Narrow_Road_Ahead", "Narrow_Bridge", "Slippery_Road", "Loose_Gravel", "Cycle_Crossing",
    "Pedestrian_Crossing", "School_Ahead", "Cattle", "Falling_Rocks", "Dangerous_Dip",
    "Hump_or_Rough_Road", "Barrier_Ahead", "Gap_in_Median", "Cross_Road", "Side_Road_Left",
    "Side_Road_Right", "Y_Intersection", "T_Intersection", "Staggered_Intersection", "Round_About",
    "Guarded_Level_Crossing", "Unguarded_Level_Crossing", "Men_at_Work", "U_Turn_Ahead", "Merge_Traffic_Ahead",
]


def load_class_names_from_yaml():
    if not DATASET_YAML.exists():
        return CLASS_NAMES
    raw = DATASET_YAML.read_text()
    match = re.search(r"names:\s*\[(.*?)\]", raw, re.DOTALL)
    if not match:
        return CLASS_NAMES
    names_str = "[" + match.group(1) + "]"
    names_str = re.sub(r"'", '"', names_str)
    try:
        return json.loads(names_str)
    except json.JSONDecodeError:
        return CLASS_NAMES


def yolo_to_pixel_bbox(x_c_norm, y_c_norm, w_norm, h_norm, img_w, img_h):
    x_c = x_c_norm * img_w
    y_c = y_c_norm * img_h
    w = w_norm * img_w
    h = h_norm * img_h
    x = int(round(x_c - w / 2))
    y = int(round(y_c - h / 2))
    width = int(round(w))
    height = int(round(h))
    return max(0, x), max(0, y), max(1, width), max(1, height)


def get_image_size(img_path):
    if Image is None:
        raise RuntimeError("PIL is required; pip install Pillow")
    with Image.open(img_path) as im:
        return im.size  # (width, height)


def main():
    class_names = load_class_names_from_yaml()
    files = []
    label_paths = sorted(LABELS_DIR.glob("*.txt"))
    for label_path in label_paths:
        stem = label_path.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = IMAGES_DIR / (stem + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue
        try:
            img_w, img_h = get_image_size(img_path)
        except Exception as e:
            print(f"Skip {img_path.name}: {e}")
            continue
        lines = label_path.read_text().strip().splitlines()
        bounding_boxes = []
        first_label = "test"
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                x_c = float(parts[1])
                y_c = float(parts[2])
                w_n = float(parts[3])
                h_n = float(parts[4])
            except (ValueError, IndexError):
                continue
            if cls_id < 0 or cls_id >= len(class_names):
                continue
            label_name = class_names[cls_id]
            if first_label == "test":
                first_label = label_name
            x, y, w, h = yolo_to_pixel_bbox(x_c, y_c, w_n, h_n, img_w, img_h)
            bounding_boxes.append({"label": label_name, "x": x, "y": y, "width": w, "height": h})
        # Path must be relative to upload root. When uploading the folder
        # yolo_dataset/images/test, files are at root (e.g. 001_001_008.jpg).
        rel_path = img_path.name
        files.append({
            "path": rel_path,
            "name": stem,
            "category": "testing",
            "label": {"type": "label", "label": first_label},
            "boundingBoxes": bounding_boxes,
        })
    out = {"version": 1, "files": files}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=4), encoding="utf-8")
    print(f"Wrote {len(files)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
