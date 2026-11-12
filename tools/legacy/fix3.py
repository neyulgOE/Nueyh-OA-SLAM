import json

# 1. Đọc file gốc của bạn (ipdetections.json)
with open('ipdetections.json', 'r') as f:
    old_data = json.load(f)

new_data_list = []

# 2. Duyệt qua từng entry và đóng gói lại vào list
for filename, detections in old_data.items():
    frame_entry = {
        "file_name": filename,
        "detections": []
    }
    
    for det in detections:
        # Chuyển đổi tên trường cho khớp hoàn toàn
        # Giả sử sink_13 dùng 'category_id' thay cho 'class'
        # và 'detection_score' thay cho 'score'
        new_det = {
            "category_id": det.get('class', 'unknown'),
            "detection_score": det.get('score', 0.0),
            "bbox": det.get('box_2d', [0, 0, 0, 0])
        }
        frame_entry["detections"].append(new_det)
    
    new_data_list.append(frame_entry)

# 3. Lưu thành file mới bắt đầu bằng dấu [
with open('ipdetections_as_list.json', 'w') as f:
    json.dump(new_data_list, f, indent=2)

print("Đã chuyển sang dạng List (Bắt đầu bằng dấu [ )")
