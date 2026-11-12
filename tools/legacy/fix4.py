import json

# Bảng ánh xạ COCO (OA-SLAM thường dùng chuẩn này)
class_to_id = { "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4, "bus": 5, "train": 6, "truck": 7, "boat": 8, "potted plant": 58, "refrigerator": 72, # Thêm các class khác nếu file của bạn có 
}

# 1. Đọc file ipdetections.json gốc
with open('/home/huyen1905/OA-SLAM/Data/realsense2.json', 'r') as f:
    old_data = json.load(f)

new_data_list = []

for filename, detections in old_data.items():
    # Tách frame_id từ tên file (ví dụ "000001.png" -> "frame_0001.png")
    frame_entry = {
        "file_name": filename,
        "detections": []
    }
    
    for det in detections:
        # Lấy ID từ bảng ánh xạ, nếu không có thì để mặc định 99
        cat_name = det.get('class', 'person')
        cat_id = class_to_id.get(cat_name, 99)
        
        new_det = {
            "category_id": int(cat_id), # Ép kiểu số nguyên (Bắt buộc để hết lỗi 302)
            "detection_score": float(det.get('score', 0.0)),
            "bbox": [float(x) for x in det.get('box_2d', [0, 0, 0, 0])]
        }
        frame_entry["detections"].append(new_det)
    
    new_data_list.append(frame_entry)

# 2. Lưu file với định dạng List chuẩn
with open('/home/huyen1905/OA-SLAM/Data/realsense2_final.json', 'w') as f:
    json.dump(new_data_list, f)

print("Đã chuyển đổi thành công! category_id hiện đã là kiểu số (Number).")

