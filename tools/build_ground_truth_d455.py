#!/usr/bin/env python3
"""
Tạo bộ Ground Truth chuẩn hóa cho cặp dữ liệu D455: dataset vs dataset2
Tuân thủ nghiêm ngặt theo MASTER PLAN v2.0 (Phase 0 & Phase 1)

Phân loại các cặp ảnh:
- POSITIVE (EASY): Lượt đi cùng chiều (dataset2: 0 - 1200 <-> dataset: 0 - 1290)
- POSITIVE (HARD): Lượt quay lại đóng vòng thực sự sau 4000 frame (dataset2: 5100 - 5575 <-> dataset: 770 - 1240)
- NEGATIVE (HARD): Cùng kiến trúc/sàn nhưng khác vị trí (dataset2: 1300 - 5000 <-> dataset)
- NEGATIVE (EASY): Các góc quay hoàn toàn khác biệt

Phân chia tập dữ liệu:
- 60% Validation Set (dùng để tune ngưỡng cho 4 Gates ở Phase 4)
- 40% Test Set (khóa kín, chỉ đánh giá 1 lần duy nhất ở Phase 7)
"""

import cv2
import numpy as np
import os
import csv
import random

DIR_REF = "/home/huyen1905/OA-SLAM/Data/dataset/images"
DIR_QRY = "/home/huyen1905/OA-SLAM/Data/dataset2/images"
OUT_DIR = "/home/huyen1905/OA-SLAM/Data/ground_truth_d455"

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)

def extract_features(img_path, orb):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    return orb.detectAndCompute(img, None)

def verify_pair(kp1, des1, kp2, des2, bf):
    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return 0, 0, 0.0
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
    if len(good) < 15:
        return len(good), 0, 0.0
        
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 2.5, 0.99)
    inliers = int(np.sum(mask)) if mask is not None else 0
    ratio = inliers / len(good) if len(good) > 0 else 0.0
    return len(good), inliers, ratio

def main():
    print("=== TẠO BỘ DỮ LIỆU GROUND TRUTH D455 THEO MASTER PLAN v2.0 ===")
    
    orb = cv2.ORB_create(nfeatures=600, fastThreshold=15)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    ref_files = sorted(os.listdir(DIR_REF))
    qry_files = sorted(os.listdir(DIR_QRY))

    pairs_record = []
    pair_id = 1

    # =========================================================================
    # 1. TẠO CẶP POSITIVE (EASY): Lượt đi (Query 0 - 1200)
    # =========================================================================
    print("1. Đang trích xuất Positive Pairs (Lượt đi - EASY)...")
    for q_idx in range(0, 1201, 25):
        q_file = qry_files[q_idx]
        q_kp, q_des = extract_features(os.path.join(DIR_QRY, q_file), orb)
        
        best_ref = None
        best_inliers = 0
        best_good = 0
        
        search_min = max(0, int(q_idx * 1.05) - 80)
        search_max = min(len(ref_files), int(q_idx * 1.05) + 80)
        
        for r_idx in range(search_min, search_max, 10):
            r_file = ref_files[r_idx]
            r_kp, r_des = extract_features(os.path.join(DIR_REF, r_file), orb)
            good, inliers, ratio = verify_pair(r_kp, r_des, q_kp, q_des, bf)
            if inliers > best_inliers and inliers >= 25 and ratio >= 0.50:
                best_inliers = inliers
                best_good = good
                best_ref = (r_idx, r_file, ratio)
                
        if best_ref is not None:
            pairs_record.append({
                "pair_id": pair_id,
                "reference_session": "dataset_day1",
                "reference_frame": best_ref[1],
                "query_session": "dataset2_day2",
                "query_frame": q_file,
                "label": "POSITIVE",
                "difficulty_tag": "EASY",
                "inliers_ransac": best_inliers,
                "notes": f"Lượt đi cùng chiều (Inliers: {best_inliers}/{best_good}, {best_ref[2]*100:.1f}%)"
            })
            pair_id += 1

    print(f"   -> Đã tạo {len(pairs_record)} cặp POSITIVE (EASY)")

    # =========================================================================
    # 2. TẠO CẶP POSITIVE (HARD): Lượt quay lại đóng vòng (Query 5100 - 5600)
    # =========================================================================
    print("2. Đang trích xuất Positive Pairs (Lượt về đóng vòng - HARD)...")
    pos_hard_count = 0
    for q_idx in range(5100, 5601, 20):
        if q_idx >= len(qry_files):
            break
        q_file = qry_files[q_idx]
        q_kp, q_des = extract_features(os.path.join(DIR_QRY, q_file), orb)
        
        best_ref = None
        best_inliers = 0
        best_good = 0
        
        for r_idx in range(700, 1300, 10):
            r_file = ref_files[r_idx]
            r_kp, r_des = extract_features(os.path.join(DIR_REF, r_file), orb)
            good, inliers, ratio = verify_pair(r_kp, r_des, q_kp, q_des, bf)
            if inliers > best_inliers and inliers >= 20 and ratio >= 0.45:
                best_inliers = inliers
                best_good = good
                best_ref = (r_idx, r_file, ratio)
                
        if best_ref is not None:
            pairs_record.append({
                "pair_id": pair_id,
                "reference_session": "dataset_day1",
                "reference_frame": best_ref[1],
                "query_session": "dataset2_day2",
                "query_frame": q_file,
                "label": "POSITIVE",
                "difficulty_tag": "HARD",
                "inliers_ransac": best_inliers,
                "notes": f"Loop Closure sau 4000 frames (Inliers: {best_inliers}/{best_good}, {best_ref[2]*100:.1f}%)"
            })
            pair_id += 1
            pos_hard_count += 1

    print(f"   -> Đã tạo {pos_hard_count} cặp POSITIVE (HARD - True Loop)")

    # =========================================================================
    # 3. TẠO CẶP NEGATIVE (HARD & EASY): Từ phân đoạn đi xa (Query 1300 - 5000)
    # =========================================================================
    print("3. Đang trích xuất Negative Pairs (Khác vị trí, bẫy cấu trúc giống)...")
    neg_hard_count = 0
    neg_easy_count = 0
    
    sampled_neg_queries = list(range(1350, 5000, 45))
    for q_idx in sampled_neg_queries:
        q_file = qry_files[q_idx]
        q_kp, q_des = extract_features(os.path.join(DIR_QRY, q_file), orb)
        
        chosen_refs = random.sample(range(0, len(ref_files), 15), 2)
        for r_idx in chosen_refs:
            r_file = ref_files[r_idx]
            r_kp, r_des = extract_features(os.path.join(DIR_REF, r_file), orb)
            good, inliers, ratio = verify_pair(r_kp, r_des, q_kp, q_des, bf)
            
            if inliers < 8:
                diff_tag = "HARD" if good >= 20 else "EASY"
                if diff_tag == "HARD":
                    neg_hard_count += 1
                else:
                    neg_easy_count += 1
                    
                pairs_record.append({
                    "pair_id": pair_id,
                    "reference_session": "dataset_day1",
                    "reference_frame": r_file,
                    "query_session": "dataset2_day2",
                    "query_frame": q_file,
                    "label": "NEGATIVE",
                    "difficulty_tag": diff_tag,
                    "inliers_ransac": inliers,
                    "notes": f"Khác vị trí thực tế (Inliers: {inliers}/{good})"
                })
                pair_id += 1

    print(f"   -> Đã tạo {neg_hard_count} cặp NEGATIVE (HARD) và {neg_easy_count} cặp NEGATIVE (EASY)")

    # =========================================================================
    # 4. CHIA THÀNH VALIDATION SET (60%) VÀ TEST SET (40%)
    # =========================================================================
    print("4. Đang phân chia Validation Set (60%) và Test Set (40%)...")
    
    groups = {}
    for r in pairs_record:
        key = (r["label"], r["difficulty_tag"])
        groups.setdefault(key, []).append(r)

    val_rows = []
    test_rows = []

    for key, items in groups.items():
        random.shuffle(items)
        split_idx = int(len(items) * 0.6)
        val_rows.extend(items[:split_idx])
        test_rows.extend(items[split_idx:])

    pairs_record.sort(key=lambda x: x["pair_id"])
    val_rows.sort(key=lambda x: x["pair_id"])
    test_rows.sort(key=lambda x: x["pair_id"])

    fieldnames = [
        "pair_id", "reference_session", "reference_frame",
        "query_session", "query_frame", "label", "difficulty_tag",
        "inliers_ransac", "notes"
    ]

    val_csv = os.path.join(OUT_DIR, "validation_pairs.csv")
    test_csv = os.path.join(OUT_DIR, "test_pairs.csv")
    full_csv = os.path.join(OUT_DIR, "all_pairs.csv")

    for path, data_list in [(full_csv, pairs_record), (val_csv, val_rows), (test_csv, test_rows)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_list)

    n_pos = sum(1 for r in pairs_record if r["label"] == "POSITIVE")
    n_pos_easy = sum(1 for r in pairs_record if r["label"] == "POSITIVE" and r["difficulty_tag"] == "EASY")
    n_pos_hard = sum(1 for r in pairs_record if r["label"] == "POSITIVE" and r["difficulty_tag"] == "HARD")
    n_neg = sum(1 for r in pairs_record if r["label"] == "NEGATIVE")
    n_neg_hard = sum(1 for r in pairs_record if r["label"] == "NEGATIVE" and r["difficulty_tag"] == "HARD")
    n_neg_easy = sum(1 for r in pairs_record if r["label"] == "NEGATIVE" and r["difficulty_tag"] == "EASY")

    print("\n=== HOÀN TẤT TẠO GROUND TRUTH ===")
    print(f"Tổng số cặp: {len(pairs_record)}")
    print(f"  - Positive pairs : {n_pos} (EASY: {n_pos_easy}, HARD: {n_pos_hard})")
    print(f"  - Negative pairs : {n_neg} (HARD: {n_neg_hard}, EASY: {n_neg_easy})")
    print()
    print(f"Tập Validation (60%): {len(val_rows)} cặp -> {val_csv}")
    print(f"Tập Test (40% - Khóa): {len(test_rows)} cặp -> {test_csv}")
    print(f"Tập Full           : {len(pairs_record)} cặp -> {full_csv}")

if __name__ == "__main__":
    main()
