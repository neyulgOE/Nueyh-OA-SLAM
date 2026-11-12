import os
import json
import torch
from ultralytics import YOLO

# 1. Khởi tạo model YOLOv8 Phân mảnh (Segmentation)
# Model sẽ tự động được tải về nếu chưa có
model = YOLO('yolov8n-seg.pt') 

# Đường dẫn tới thư mục chứa ảnh RGB của dataset (ví dụ: nuscenes)
image_folder = "/home/huyen1905/OA-SLAM/Data/dataset5/images/"
output_json_path = "/home/huyen1905/OA-SLAM/Data/realsense55_final2.json"

all_frames = []
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

print(f"Đang xử lý {len(image_files)} ảnh...")

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    results = model(img_path, verbose=False)
    
    frame_detections = []
    if results[0].boxes is not None and results[0].masks is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        
        for i in range(len(boxes)):
            if scores[i] > 0.5:
                det = {
                    "category_id": int(classes[i]),
                    "detection_score": float(scores[i]), # C++ bắt buộc tên là detection_score
                    "bbox": [float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])],
                    "mask_polygon": results[0].masks.xy[i].tolist() # Tọa độ mask
                }
                frame_detections.append(det)
                
    # Gói vào chuẩn C++ mong muốn
    all_frames.append({
        "file_name": img_file,
        "detections": frame_detections
    })

with open(output_json_path, 'w') as f:
    json.dump(all_frames, f)

print("Hoàn thành sinh file JSON chuẩn!")
