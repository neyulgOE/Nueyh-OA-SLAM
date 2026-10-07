#!/usr/bin/env python3
"""
Convert detection JSON files to OA-SLAM standard format.
Supports:
  - Input as Dict { "000001.png": [...] } or List of frames
  - Bounding box auto-detection: [x1, y1, x2, y2] -> [x, y, w, h]
  - Class mapping: string class names ("person", "chair", ...) to COCO integer IDs
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

CLASS_NAME_TO_ID = {name: i for i, name in enumerate(COCO_CLASSES)}


def parse_bbox(raw_box: Any, box_format: str = "auto") -> List[float]:
    """Ensure box is [x, y, w, h] in absolute pixel coordinates."""
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
        return [0.0, 0.0, 0.0, 0.0]

    b0, b1, b2, b3 = float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3])

    if box_format == "xywh":
        return [b0, b1, b2, b3]
    elif box_format == "xyxy":
        return [b0, b1, max(0.0, b2 - b0), max(0.0, b3 - b1)]
    else:  # auto
        # If b2 > b0 and b3 > b1, and b2, b3 could be x2, y2
        if b2 > b0 and b3 > b1 and (b2 - b0) > 0 and (b3 - b1) > 0:
            # Likely [x1, y1, x2, y2]
            return [b0, b1, b2 - b0, b3 - b1]
        return [b0, b1, b2, b3]


def parse_category(raw_cat: Any) -> int:
    """Parse class name or ID into integer COCO category ID."""
    if isinstance(raw_cat, int):
        return raw_cat
    if isinstance(raw_cat, str):
        if raw_cat.isdigit():
            return int(raw_cat)
        normalized = raw_cat.strip().lower()
        if normalized in CLASS_NAME_TO_ID:
            return CLASS_NAME_TO_ID[normalized]
    return -1


def convert_detections(input_path: str, output_path: str, box_format: str = "auto", min_score: float = 0.25):
    with open(input_path, 'r') as f:
        data = json.load(f)

    converted_frames = []

    # Case A: Dict format { "000000.png": [ {...}, ... ], ... }
    if isinstance(data, dict):
        for filename, raw_dets in sorted(data.items()):
            frame_entry = {"file_name": os.path.basename(filename), "detections": []}
            if isinstance(raw_dets, list):
                for d in raw_dets:
                    score = float(d.get("score", d.get("detection_score", d.get("confidence", 0.0))))
                    if score < min_score:
                        continue
                    cat = d.get("class", d.get("category_id", d.get("label", -1)))
                    cat_id = parse_category(cat)
                    if cat_id < 0:
                        continue
                    box = d.get("bbox", d.get("box_2d", d.get("box", [0, 0, 0, 0])))
                    bbox_xywh = parse_bbox(box, box_format)
                    frame_entry["detections"].append({
                        "category_id": int(cat_id),
                        "detection_score": float(round(score, 4)),
                        "bbox": [round(v, 2) for v in bbox_xywh]
                    })
            converted_frames.append(frame_entry)

    # Case B: List format [ {"file_name": ..., "detections": [...] }, ... ]
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            fname = item.get("file_name", item.get("filename", item.get("image", "")))
            frame_entry = {"file_name": os.path.basename(fname), "detections": []}
            raw_dets = item.get("detections", item.get("boxes", []))
            for d in raw_dets:
                score = float(d.get("score", d.get("detection_score", d.get("confidence", 0.0))))
                if score < min_score:
                    continue
                cat = d.get("class", d.get("category_id", d.get("label", -1)))
                cat_id = parse_category(cat)
                if cat_id < 0:
                    continue
                box = d.get("bbox", d.get("box_2d", d.get("box", [0, 0, 0, 0])))
                bbox_xywh = parse_bbox(box, box_format)
                frame_entry["detections"].append({
                    "category_id": int(cat_id),
                    "detection_score": float(round(score, 4)),
                    "bbox": [round(v, 2) for v in bbox_xywh]
                })
            converted_frames.append(frame_entry)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(converted_frames, f, indent=2)

    total_dets = sum(len(f["detections"]) for f in converted_frames)
    print(f"[OK] Converted {len(converted_frames)} frames with {total_dets} detections to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert detections to OA-SLAM format")
    parser.add_argument("-i", "--input", required=True, help="Path to input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Path to output JSON file")
    parser.add_argument("--format", choices=["auto", "xyxy", "xywh"], default="auto", help="BBox format")
    parser.add_argument("--min-score", type=float, default=0.25, help="Minimum confidence threshold")
    args = parser.parse_args()

    convert_detections(args.input, args.output, args.format, args.min_score)
