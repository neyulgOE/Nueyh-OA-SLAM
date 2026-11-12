#!/usr/bin/env python3
"""
Phase 2: Global Place Recognition Benchmark - Baseline 1 (ORB Matching)
Dự án: Long-term Visual Place Recognition for OA-SLAM

Mục tiêu:
1. Xây dựng Database gồm các Reference KeyFrames từ Data/dataset (1360 frames).
2. Đánh giá khả năng tìm kiếm Top-K candidate của ORB trên tập Validation:
   - Recall@1, Recall@5, Recall@10, Recall@20 cho POSITIVE EASY (Lượt đi).
   - Recall@1, Recall@5, Recall@10, Recall@20 cho POSITIVE HARD (Loop Closure sau 4000 frames).
3. Đánh giá tỉ lệ từ chối (Rejection Rate) trên tập NEGATIVE.
4. Đo thời gian xử lý (Latency) trung bình mỗi truy vấn.
"""

import cv2
import numpy as np
import os
import csv
import time

DIR_REF = "/home/huyen1905/OA-SLAM/Data/dataset/images"
DIR_QRY = "/home/huyen1905/OA-SLAM/Data/dataset2/images"
VAL_CSV = "/home/huyen1905/OA-SLAM/Data/ground_truth_d455/validation_pairs.csv"

# Trong SLAM, KeyFrames được tạo cách quãng (~5-10 frames/KeyFrame)
KF_STEP = 5 
TOLERANCE_FRAMES = 15  # Coi là đúng nếu tìm được frame trong khoảng +-15 frames của Ground Truth

def extract_orb(img_path, orb):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, des = orb.detectAndCompute(img, None)
    return des

def main():
    print("=================================================================")
    print("   PHASE 2 BENCHMARK: BASELINE 1 — ORB GLOBAL RETRIEVAL         ")
    print("=================================================================")
    
    orb = cv2.ORB_create(nfeatures=1000, fastThreshold=15)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # 1. Xây dựng Reference Database
    ref_files = sorted(os.listdir(DIR_REF))
    db_keyframes = []
    
    print(f"\n1. Đang lập chỉ mục Reference Database (Bước nhảy {KF_STEP} frames)...")
    t0 = time.time()
    for idx in range(0, len(ref_files), KF_STEP):
        fname = ref_files[idx]
        des = extract_orb(os.path.join(DIR_REF, fname), orb)
        if des is not None:
            db_keyframes.append((idx, fname, des))
    print(f"   -> Đã index {len(db_keyframes)} Reference KeyFrames trong {time.time()-t0:.2f}s")

    # 2. Đọc tập Validation Pairs
    with open(VAL_CSV, "r", encoding="utf-8") as f:
        val_pairs = list(csv.DictReader(f))

    pos_easy = [r for r in val_pairs if r["label"] == "POSITIVE" and r["difficulty_tag"] == "EASY"]
    pos_hard = [r for r in val_pairs if r["label"] == "POSITIVE" and r["difficulty_tag"] == "HARD"]
    neg_pairs = [r for r in val_pairs if r["label"] == "NEGATIVE"]

    print(f"\n2. Tập Validation: {len(val_pairs)} cặp")
    print(f"   - Positive EASY (Lượt đi)           : {len(pos_easy)} queries")
    print(f"   - Positive HARD (Loop sau 4000 frame): {len(pos_hard)} queries")
    print(f"   - Negative (Đi xa)                  : {len(neg_pairs)} queries")

    # Hàm truy vấn Top-K cho 1 ảnh
    def query_top_k(q_des, top_k=20):
        scores = []
        for r_idx, r_fname, r_des in db_keyframes:
            matches = bf.knnMatch(q_des, r_des, k=2)
            good = [m for m, n in matches if len(matches) > 0 and m.distance < 0.75 * n.distance]
            scores.append((len(good), r_idx, r_fname))
        # Sắp xếp giảm dần theo số good matches
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

    # 3. Đánh giá trên Positive Pairs
    def evaluate_positive(pairs, group_name):
        print(f"\n--- Đang đánh giá {group_name} ({len(pairs)} queries) ---")
        recalls = {1: 0, 5: 0, 10: 0, 20: 0}
        latencies = []

        for p in pairs:
            q_file = p["query_frame"]
            target_r_file = p["reference_frame"]
            target_r_idx = int(target_r_file.split(".")[0])

            q_des = extract_orb(os.path.join(DIR_QRY, q_file), orb)
            if q_des is None:
                continue

            t_start = time.time()
            top_results = query_top_k(q_des, top_k=20)
            latencies.append((time.time() - t_start) * 1000)

            # Kiểm tra xem target có nằm trong Top-K không
            retrieved_indices = [r[1] for r in top_results]
            
            for k in [1, 5, 10, 20]:
                subset = retrieved_indices[:k]
                # Coi là hit nếu tìm thấy bất kỳ frame nào cách target <= TOLERANCE_FRAMES
                hit = any(abs(idx - target_r_idx) <= TOLERANCE_FRAMES for idx in subset)
                if hit:
                    recalls[k] += 1

        n = len(pairs)
        r1 = (recalls[1] / n) * 100 if n > 0 else 0
        r5 = (recalls[5] / n) * 100 if n > 0 else 0
        r10 = (recalls[10] / n) * 100 if n > 0 else 0
        r20 = (recalls[20] / n) * 100 if n > 0 else 0
        avg_lat = np.mean(latencies) if latencies else 0

        print(f"  Recall@1  : {r1:5.1f}% ({recalls[1]}/{n})")
        print(f"  Recall@5  : {r5:5.1f}% ({recalls[5]}/{n})")
        print(f"  Recall@10 : {r10:5.1f}% ({recalls[10]}/{n})")
        print(f"  Recall@20 : {r20:5.1f}% ({recalls[20]}/{n})")
        print(f"  Độ trễ TB : {avg_lat:.1f} ms/query")
        return {"r1": r1, "r5": r5, "r10": r10, "r20": r20, "lat": avg_lat}

    res_easy = evaluate_positive(pos_easy, "POSITIVE - EASY (Lượt đi)")
    res_hard = evaluate_positive(pos_hard, "POSITIVE - HARD (Loop sau 4000 frames)")

    # 4. Đánh giá trên Negative Pairs (Kiểm tra điểm match cao nhất có bị false alarm không)
    print(f"\n--- Đang đánh giá NEGATIVE ({len(neg_pairs)} queries) ---")
    neg_top_scores = []
    for p in neg_pairs[:30]:  # Lấy mẫu 30 negative để đo nhanh
        q_file = p["query_frame"]
        q_des = extract_orb(os.path.join(DIR_QRY, q_file), orb)
        if q_des is not None:
            top_res = query_top_k(q_des, top_k=1)
            neg_top_scores.append(top_res[0][0]) # Số match cao nhất

    avg_neg_score = np.mean(neg_top_scores) if neg_top_scores else 0
    max_neg_score = np.max(neg_top_scores) if neg_top_scores else 0
    print(f"  Điểm matches cao nhất trên Negative TB : {avg_neg_score:.1f} matches")
    print(f"  Điểm matches cao nhất từng gặp        : {max_neg_score} matches")

    # 5. Bảng tổng kết Phase 2 Baseline
    print("\n=================================================================")
    print("                    KẾT QUẢ PHASE 2 BASELINE (ORB)               ")
    print("=================================================================")
    print(f"| Kịch bản | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Độ trễ |")
    print(f"|---|:---:|:---:|:---:|:---:|:---:|")
    print(f"| Positive (EASY - Lượt đi) | {res_easy['r1']:.1f}% | {res_easy['r5']:.1f}% | {res_easy['r10']:.1f}% | {res_easy['r20']:.1f}% | {res_easy['lat']:.1f} ms |")
    print(f"| Positive (HARD - Loop về) | {res_hard['r1']:.1f}% | {res_hard['r5']:.1f}% | {res_hard['r10']:.1f}% | {res_hard['r20']:.1f}% | {res_hard['lat']:.1f} ms |")
    print("=================================================================")

if __name__ == "__main__":
    main()

