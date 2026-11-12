import json

# Đọc file lỗi
with open('ipdetections.json', 'r') as f:
    old_data = json.load(f)

new_data = {}
# Duyệt qua từng entry, giả sử tên file là "000000.png" thì lấy "0"
for filename, detections in old_data.items():
    # Lấy phần số và bỏ các số 0 ở đầu (ví dụ "000005.png" -> "5")
    frame_id = str(int(filename.split('.')[0]))
    new_data[frame_id] = detections

# Lưu lại file mới chuẩn cho OA-SLAM
with open('ipdetections_fixed.json', 'w') as f:
    json.dump(new_data, f, indent=2)
print("Đã fix xong! Hãy dùng file ipdetections_fixed.json để chạy OA-SLAM.")
