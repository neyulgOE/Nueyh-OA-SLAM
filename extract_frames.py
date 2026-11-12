#!/usr/bin/env python3
"""
Phase 1: Frame Extractor & Motion Blur Filter for iPhone 15 Plus Video
Dự án: Long-term Visual Place Recognition for OA-SLAM

Chức năng:
1. Đọc video gốc (4K 30fps).
2. Lấy mẫu theo tần số chỉ định (mặc định 1.5 FPS để bao phủ không gian, tránh thừa dữ liệu).
3. Đánh giá độ sắc nét bằng phương sai toán tử Laplacian (Laplacian Variance).
4. Phân loại frame: KEEP (sắc nét) hoặc DROP_BLUR (mờ do chuyển động).
5. Xuất ảnh và file metadata.csv chuẩn hóa.
"""

import cv2
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def calculate_blur_score(image_gray):
    """
    Tính độ sắc nét dựa trên phương sai của toán tử Laplacian.
    Điểm càng cao -> ảnh càng sắc nét.
    Ảnh bị mờ chuyển động thường có điểm < 100 (tùy độ phân giải).
    """
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    score = laplacian.var()
    return float(score)


def process_video(video_path, output_dir, target_fps=1.5, blur_threshold=100.0, max_dimension=1920):
    """
    Trích xuất và lọc frame từ video.
    max_dimension: Resize ảnh lưu trữ về Full HD (1920x1080) để tối ưu dung lượng và tốc độ benchmark VPR,
                   nhưng giữ nguyên tỉ lệ khung hình của 4K gốc.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở file video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / orig_fps if orig_fps > 0 else 0

    print(f"=== XỬ LÝ VIDEO: {video_path.name} ===")
    print(f"Độ dài: {duration_sec:.2f} giây | FPS gốc: {orig_fps:.2f} | Tổng frames: {total_frames}")
    print(f"Target sampling FPS: {target_fps} | Ngưỡng lọc mờ (Laplacian Var): {blur_threshold}")

    # Tính khoảng cách bước nhảy giữa các frame cần lấy mẫu
    frame_step = int(round(orig_fps / target_fps))
    if frame_step < 1:
        frame_step = 1

    metadata = []
    saved_count = 0
    dropped_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            timestamp_sec = frame_idx / orig_fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = calculate_blur_score(gray)

            status = "KEEP" if blur_score >= blur_threshold else "DROP_BLUR"
            saved_filename = ""

            if status == "KEEP":
                # Resize ảnh chuẩn (ví dụ chiều rộng 1920, giữ nguyên aspect ratio)
                h, w = frame.shape[:2]
                if w > max_dimension:
                    scale = max_dimension / float(w)
                    new_w, new_h = max_dimension, int(h * scale)
                    frame_to_save = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    frame_to_save = frame

                saved_filename = f"frame_{saved_count:06d}.jpg"
                save_path = images_dir / saved_filename
                # Lưu JPEG chất lượng cao (quality 95)
                cv2.imwrite(str(save_path), frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1
            else:
                dropped_count += 1

            metadata.append({
                "frame_idx_orig": frame_idx,
                "timestamp_sec": round(timestamp_sec, 3),
                "saved_filename": saved_filename,
                "blur_score": round(blur_score, 2),
                "status": status
            })

            if len(metadata) % 50 == 0:
                print(f"  -> Đã duyệt {frame_idx}/{total_frames} frames... Đã lưu: {saved_count}, Bỏ mờ: {dropped_count}")

        frame_idx += 1

    cap.release()

    # Xuất metadata CSV
    df = pd.DataFrame(metadata)
    csv_path = output_dir / "metadata.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== HOÀN TẤT ===")
    print(f"Tổng frame kiểm tra: {len(metadata)}")
    print(f"Frame đạt chuẩn (đã lưu): {saved_count}")
    print(f"Frame bị loại do nhòe mờ: {dropped_count}")
    print(f"File metadata: {csv_path}")
    print(f"Thư mục ảnh: {images_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and filter frames from video for VPR Dataset")
    parser.add_argument("--video", type=str, required=True, help="Đường dẫn file video đầu vào (.mov hoặc .mp4)")
    parser.add_argument("--output", type=str, required=True, help="Thư mục xuất dữ liệu (processed/session_X)")
    parser.add_argument("--fps", type=float, default=1.5, help="Tần số lấy mẫu ảnh (mặc định: 1.5 FPS)")
    parser.add_argument("--threshold", type=float, default=100.0, help="Ngưỡng Laplacian Variance lọc mờ (mặc định: 100.0)")
    parser.add_argument("--max_dim", type=int, default=1920, help="Kích thước chiều rộng tối đa (mặc định: 1920)")

    args = parser.parse_args()
    process_video(args.video, args.output, args.fps, args.threshold, args.max_dim)

