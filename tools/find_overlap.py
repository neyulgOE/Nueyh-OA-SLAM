#!/usr/bin/env python3
"""
Find Overlap Segments between dataset (1360 frames) and dataset2 (5636 frames)
Dự án: Long-term Visual Place Recognition for OA-SLAM

Quy trình:
1. Lấy mẫu dataset cách mỗi 10 frames (136 frames).
2. Lấy mẫu dataset2 cách mỗi 25 frames (225 frames).
3. Trích xuất ORB features (500 features/frame).
4. Khớp đặc trưng thô để tìm các cặp tiềm năng (Candidate Pairs).
5. Xác minh hình học bằng Fundamental Matrix RANSAC (Inliers >= 20, ratio >= 40%).
6. Nhóm thành các dải phân đoạn trùng lặp (Overlap Intervals).
"""

import cv2
import numpy as np
import os
import time

DIR_REF = "/home/huyen1905/OA-SLAM/Data/dataset/images"
DIR_QRY = "/home/huyen1905/OA-SLAM/Data/dataset2/images"

STEP_REF = 10  # Lấy mẫu dataset mỗi 10 frame (136 frames)
STEP_QRY = 25  # Lấy mẫu dataset2 mỗi 25 frame (225 frames)

def main():
    ref_files = sorted(os.listdir(DIR_REF))
    qry_files = sorted(os.listdir(DIR_QRY))
    
    ref_sampled = [(i, ref_files[i]) for i in range(0, len(ref_files), STEP_REF)]
    qry_sampled = [(i, qry_files[i]) for i in range(0, len(qry_files), STEP_QRY)]
    
    print(f"=== BẮT ĐẦU QUÉT TÌM ĐOẠN TRÙNG LẶP (OVERLAP DISCOVERY) ===")
    print(f"Dataset Reference (dataset) : {len(ref_files)} frames -> Lấy mẫu {len(ref_sampled)} frames (bước {STEP_REF})")
    print(f"Dataset Query     (dataset2): {len(qry_files)} frames -> Lấy mẫu {len(qry_sampled)} frames (bước {STEP_QRY})")
    print()

    orb = cv2.ORB_create(nfeatures=500, fastThreshold=15)
    
    # 1. Trích xuất đặc trưng cho Reference (dataset)
    print("1. Đang trích xuất đặc trưng cho Reference frames...")
    ref_data = []
    t0 = time.time()
    for idx, fname in ref_sampled:
        img = cv2.imread(os.path.join(DIR_REF, fname), cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None)
        if des is not None and len(kp) >= 50:
            ref_data.append((idx, fname, kp, des))
    print(f"   -> Đã trích xuất {len(ref_data)} frames ({time.time()-t0:.2f}s)")

    # 2. Trích xuất đặc trưng cho Query (dataset2)
    print("2. Đang trích xuất đặc trưng cho Query frames...")
    qry_data = []
    t0 = time.time()
    for idx, fname in qry_sampled:
        img = cv2.imread(os.path.join(DIR_QRY, fname), cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None)
        if des is not None and len(kp) >= 50:
            qry_data.append((idx, fname, kp, des))
    print(f"   -> Đã trích xuất {len(qry_data)} frames ({time.time()-t0:.2f}s)")
    print()

    # 3. Khớp và Xác minh hình học
    print("3. Đang quét tìm các cặp khớp hình học (RANSAC verification)...")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    verified_matches = []
    t0 = time.time()
    
    for q_idx, q_fname, q_kp, q_des in qry_data:
        for r_idx, r_fname, r_kp, r_des in ref_data:
            matches = bf.knnMatch(r_des, q_des, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            
            # Ngưỡng sơ loại: tối thiểu 25 good matches
            if len(good) >= 25:
                pts1 = np.float32([r_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                pts2 = np.float32([q_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                
                F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 2.5, 0.99)
                if mask is not None:
                    inliers = int(np.sum(mask))
                    ratio = inliers / len(good)
                    if inliers >= 20 and ratio >= 0.40:
                        verified_matches.append({
                            "ref_idx": r_idx,
                            "ref_file": r_fname,
                            "qry_idx": q_idx,
                            "qry_file": q_fname,
                            "inliers": inliers,
                            "good": len(good),
                            "ratio": round(ratio * 100, 1)
                        })

    print(f"   -> Hoàn thành quét trong {time.time()-t0:.2f}s!")
    print(f"   -> Tìm thấy tổng cộng {len(verified_matches)} cặp frame trùng lặp được xác minh hình học.")
    print()

    if not verified_matches:
        print("Không tìm thấy đoạn giao nhau nào khác ngoài frame 0.")
        return

    # Sắp xếp theo query index
    verified_matches.sort(key=lambda x: (x["qry_idx"], -x["inliers"]))

    print(f"| Query (dataset2) | Ref (dataset) | Inliers / Matches | Tỉ lệ Inliers |")
    print(f"|:---:|:---:|:---:|:---:|")
    
    # In danh sách các cặp nổi bật
    prev_q = -1
    for m in verified_matches:
        # Chỉ in 1 ref tốt nhất cho mỗi query frame để bảng gọn
        if m["qry_idx"] != prev_q:
            print(f"| Frame {m['qry_idx']:04d} ({m['qry_file']}) | Frame {m['ref_idx']:04d} ({m['ref_file']}) | {m['inliers']:3d} / {m['good']:3d} | {m['ratio']}% |")
            prev_q = m["qry_idx"]

if __name__ == "__main__":
    main()

