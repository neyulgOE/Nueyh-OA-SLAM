import json

input_file = '/home/huyen1905/OA-SLAM/Data/nuscenes_detections_yolov5.json'
output_file = '/home/huyen1905/OA-SLAM/Data/nuscenes_detections_fixed.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for frame in data:
    for det in frame.get('detections', []):
        if 'detection_score' not in det or det['detection_score'] is None:
            det['detection_score'] = 1.0

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Đã sửa xong → {output_file}")
