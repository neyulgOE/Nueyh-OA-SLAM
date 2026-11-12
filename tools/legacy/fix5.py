import json

# Bảng ánh xạ COCO (OA-SLAM thường dùng chuẩn này)
class_to_id = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79
}

# 1. Đọc file ipdetections.json gốc
with open('/home/huyen1905/OA-SLAM/Data/realsense5.json', 'r') as f:
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
with open('/home/huyen1905/OA-SLAM/Data/realsense55_final.json', 'w') as f:
    json.dump(new_data_list, f)

print("Đã chuyển đổi thành công! category_id hiện đã là kiểu số (Number).")

