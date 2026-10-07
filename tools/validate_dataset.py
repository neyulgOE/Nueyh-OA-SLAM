#!/usr/bin/env python3
"""
Validate dataset integrity before running OA-SLAM.
Checks:
  - Camera config YAML
  - Image sequence (folder with rgb.txt, or list of images)
  - Detection JSON (checks if file_names match the images)
  - Categories to ignore file
"""

import os
import sys
import json
import argparse


def validate_dataset(camera_yaml: str, img_seq: str, detections_json: str = None, ignore_file: str = None):
    print("=== [OA-SLAM Dataset Validation] ===")
    errors = []
    warnings = []

    # 1. Camera YAML
    if not os.path.isfile(camera_yaml):
        errors.append(f"Camera config YAML not found: {camera_yaml}")
    else:
        print(f"[OK] Camera YAML found: {camera_yaml}")

    # 2. Image sequence / folder
    image_names = set()
    if os.path.isdir(img_seq):
        rgb_txt = os.path.join(img_seq, "rgb.txt")
        if os.path.isfile(rgb_txt):
            with open(rgb_txt) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            image_names.add(os.path.basename(parts[1]))
            print(f"[OK] Found rgb.txt with {len(image_names)} images in {img_seq}")
        else:
            # list images in folder
            imgs = [f for f in os.listdir(img_seq) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_names = set(imgs)
            print(f"[OK] Found {len(image_names)} image files in {img_seq}")
    elif os.path.isfile(img_seq):
        with open(img_seq) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    fname = parts[-1]
                    image_names.add(os.path.basename(fname))
        print(f"[OK] Found list file with {len(image_names)} images: {img_seq}")
    else:
        errors.append(f"Image sequence path not found: {img_seq}")

    # 3. Detections JSON
    if detections_json:
        if not os.path.isfile(detections_json):
            errors.append(f"Detections JSON file not found: {detections_json}")
        else:
            try:
                with open(detections_json) as f:
                    dets = json.load(f)
                if not isinstance(dets, list):
                    errors.append("Detections JSON must be a list of frame objects: [ { 'file_name': ..., 'detections': [...] } ]")
                else:
                    det_names = set()
                    total_objs = 0
                    for frame in dets:
                        fname = frame.get("file_name", "")
                        det_names.add(os.path.basename(fname))
                        total_objs += len(frame.get("detections", []))
                    print(f"[OK] Detections JSON contains {len(det_names)} frames, {total_objs} detections")

                    if image_names:
                        matched = image_names.intersection(det_names)
                        pct = (len(matched) / len(image_names)) * 100
                        if pct < 50:
                            warnings.append(f"Only {len(matched)}/{len(image_names)} ({pct:.1f}%) images matched between rgb list and detections JSON. Check filename padding (e.g., '000001.png' vs '1.png')!")
                        else:
                            print(f"[OK] Image-Detection matching rate: {pct:.1f}% ({len(matched)}/{len(image_names)} matched)")
            except Exception as e:
                errors.append(f"Failed to parse detections JSON: {e}")

    # 4. Ignore categories file
    if ignore_file:
        if not os.path.isfile(ignore_file):
            warnings.append(f"Ignore categories file not found: {ignore_file}. (OA-SLAM will run with 0 ignored classes)")
        else:
            with open(ignore_file) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            print(f"[OK] Categories to ignore: {len(lines)} classes")

    print("\n=== [Summary] ===")
    if warnings:
        for w in warnings:
            print(f"[WARNING] {w}")
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        return False
    else:
        print("[SUCCESS] Dataset validation passed! Ready to run OA-SLAM.")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset before running OA-SLAM")
    parser.add_argument("--camera", required=True, help="Path to camera yaml")
    parser.add_argument("--seq", required=True, help="Path to image folder or list txt")
    parser.add_argument("--detections", required=False, help="Path to detections json")
    parser.add_argument("--ignore", required=False, help="Path to ignore classes txt")
    args = parser.parse_args()

    success = validate_dataset(args.camera, args.seq, args.detections, args.ignore)
    sys.exit(0 if success else 1)
