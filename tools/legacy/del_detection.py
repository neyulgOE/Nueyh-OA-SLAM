import json
import os

input_path="/home/huyen1905/OA-SLAM/Data/nuscenes_detections_yolov5.json"

if not os.path.exists(input_path):
    print(f"Lỗi: Không tìm thấy file tại {input_path}")
else:
    with open(input_path, 'r') as f:
        raw_data = json.load(f)

    clean_data = {}

    # Nếu raw_data là một List (Dạng: [{...}, {...}])
    if isinstance(raw_data, list):
        print("Phát hiện định dạng List, đang chuyển sang Dictionary...")
        for entry in raw_data:
            # Lấy key từ frame_id hoặc tên file nếu có
            key = str(entry.get("frame_id", "unknown"))
            detections = entry.get("detections", [])
            clean_data[key] = detections
    else:
        # Nếu đã là Dictionary (Dạng: {"frame_1": [...]})
        clean_data = raw_data

    # Làm sạch các giá trị null và ép kiểu về số (Number)
    for frame_key in clean_data:
        for det in clean_data[frame_key]:
            # Sửa tọa độ box_2d: thay null bằng 0.0 và ép kiểu float
            if "box_2d" in det:
                det["box_2d"] = [float(x) if x is not None else 0.0 for x in det["box_2d"]]
            
            # Sửa điểm số score: thay null bằng 0.0
            if "score" in det:
                det["score"] = float(det["score"]) if det["score"] is not None else 0.0

    # Lưu lại file với định dạng Object {} chuẩn
    with open(input_path, 'w') as f:
        json.dump(clean_data, f, indent=2)
    
    print("HOÀN THÀNH: File đã được chuyển sang Dictionary và xóa sạch giá trị null!")
