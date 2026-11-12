import cv2
import os
from glob import glob

# --- CẤU HÌNH ---
INPUT_DIR = "/home/huyen1905/OA-SLAM/Data/demoVideo/stuttgart_00"
OUTPUT_DIR = "/home/huyen1905/OA-SLAM/Data/demoVideo/stuttgart_00_resize1"
WIDTH = 640
HEIGHT = 320

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lấy danh sách ảnh (.png hoặc .jpg)
image_files = glob(os.path.join(INPUT_DIR, "*.png")) + glob(os.path.join(INPUT_DIR, "*.jpg"))

print(f"Bắt đầu resize {len(image_files)} ảnh...")

for img_path in image_files:
    # Đọc ảnh gốc (2048x1024)
    img = cv2.imread(img_path)
    
    if img is None:
        continue

    # Resize về 640x480
    # INTER_AREA thường cho chất lượng tốt nhất khi thu nhỏ ảnh
    resized_img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    
    # Lưu ảnh
    file_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), resized_img)

print(f"Hoàn thành! Ảnh được lưu tại: {OUTPUT_DIR}")
