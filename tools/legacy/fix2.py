import json

# Đọc file gốc (file đang bị lỗi array)
with open('/home/huyen1905/OA-SLAM/ipdetections.json', 'r') as f:
    data = json.load(f)

fixed_object = {}

# Trường hợp 1: Nếu data đang là một list các object [{filename: ...}, ...]
if isinstance(data, list):
    for i, item in enumerate(data):
        # Nếu item là dict và có key là tên file (như bạn đã gửi)
        if isinstance(item, dict):
            for filename, detections in item.items():
                try:
                    frame_id = str(int(filename.split('.')[0]))
                    fixed_object[frame_id] = detections
                except:
                    continue
        else:
            # Nếu item trực tiếp là list detections
            fixed_object[str(i)] = item

# Trường hợp 2: Nếu data đã là dict nhưng key vẫn là "000000.png"
elif isinstance(data, dict):
    for filename, detections in data.items():
        try:
            frame_id = str(int(filename.split('.')[0]))
            fixed_object[frame_id] = detections
        except:
            # Nếu key đã là số "0", "1" thì giữ nguyên
            fixed_object[filename] = detections

# Ghi đè lại file fixed với cấu hình chuẩn Object
with open('/home/huyen1905/OA-SLAM/ipdetections_fixed.json', 'w') as f:
    json.dump(fixed_object, f, indent=2)

print("Đã chuyển đổi thành công sang định dạng Object { 'ID': [...] }")
