#!/usr/bin/env python3
"""
Benchmark đối chứng: SIFT + Dynamic Mask vs ORB + Dynamic Mask vs SuperPoint + LightGlue + Mask
Trên toàn bộ 217 cặp Ground Truth giữa Data/dataset (Day 1) và Data/dataset2 (Day 2).
"""

import os
import csv
import json
import time
import cv2
import numpy as np
DIR_REF = "/home/huyen1905/OA-SLAM/Data/dataset/images"
DIR_QRY = "/home/huyen1905/OA-SLAM/Data/dataset2/images"
JSON_REF = "/home/huyen1905/OA-SLAM/Data/d455_final.json"
JSON_QRY = "/home/huyen1905/OA-SLAM/Data/realsense25_final.json"
ALL_CSV = "/home/huyen1905/OA-SLAM/Data/ground_truth_d455/all_pairs.csv"

K_D455 = np.array([
    [393.7687, 0.0, 315.1953],
    [0.0, 393.7962, 239.3341],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DYNAMIC_CLASSES = {0, 1, 2, 3, 24, 26, 28}


def load_detections_map(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item.get("file_name"): item.get("detections", []) for item in data}


def generate_static_mask(h, w, detections, min_score=0.3):
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for det in detections:
        if det.get("category_id") in DYNAMIC_CLASSES and det.get("detection_score", 0.0) >= min_score:
            bbox = det.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 0
    return mask


def eval_matcher(pairs, dets_ref, dets_qry, mode="SIFT"):
    if mode == "SIFT":
        extractor = cv2.SIFT_create(nfeatures=1500)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        ratio_thresh = 0.75
    elif mode == "ORB":
        extractor = cv2.ORB_create(nfeatures=2000, fastThreshold=8)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        ratio_thresh = 0.75
    else:
        raise ValueError(mode)

    stats = {
        "POSITIVE_EASY": {"inliers": [], "ratios": [], "verified": 0, "total": 0},
        "POSITIVE_HARD": {"inliers": [], "ratios": [], "verified": 0, "total": 0},
        "NEGATIVE_EASY": {"inliers": [], "ratios": [], "verified": 0, "total": 0},
    }
    latencies = []

    for p in pairs:
        r_file = p["reference_frame"]
        q_file = p["query_frame"]
        lbl = p["label"] + "_" + p["difficulty_tag"]

        im1 = cv2.imread(os.path.join(DIR_REF, r_file), cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(os.path.join(DIR_QRY, q_file), cv2.IMREAD_GRAYSCALE)
        if im1 is None or im2 is None:
            continue

        h, w = im1.shape[:2]
        m1 = generate_static_mask(h, w, dets_ref.get(r_file, []))
        m2 = generate_static_mask(h, w, dets_qry.get(q_file, []))

        t0 = time.time()
        kp1, des1 = extractor.detectAndCompute(im1, m1)
        kp2, des2 = extractor.detectAndCompute(im2, m2)

        n_inliers = 0
        ratio = 0.0
        verified = False

        if des1 is not None and des2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
            matches = bf.knnMatch(des1, des2, k=2)
            good = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < ratio_thresh * n.distance]
            if len(good) >= 8:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
                E, mask_E = cv2.findEssentialMat(
                    pts1, pts2, K_D455, method=cv2.RANSAC, prob=0.999, threshold=1.8
                )
                if E is not None and mask_E is not None:
                    idx_in = np.where(mask_E.ravel() == 1)[0]
                    n_inliers = len(idx_in)
                    ratio = (n_inliers / len(good)) * 100.0 if len(good) > 0 else 0.0

                    # Áp dụng cùng bộ lọc Gate 1 + Gate 2 (giống CascadingVerifier)
                    if n_inliers >= 35 and (n_inliers / len(good)) >= 0.40:
                        pts1_in = pts1[idx_in]
                        pts2_in = pts2[idx_in]
                        num_pos, _, _, _ = cv2.recoverPose(E, pts1_in, pts2_in, K_D455)
                        if (num_pos / n_inliers) >= 0.50:
                            # Gate 2: Spatial grid 3x3 & Hull area
                            def check_spatial(pts):
                                grid = np.zeros((3, 3), dtype=np.int32)
                                for x, y in pts:
                                    grid[min(2, int(y / (h / 3))), min(2, int(x / (w / 3)))] += 1
                                occ = int(np.sum(grid >= 2))
                                hull = cv2.contourArea(cv2.convexHull(pts)) / (w * h)
                                return occ >= 4 and hull >= 0.15 and np.std(pts[:, 0]) >= 55.0 and np.std(pts[:, 1]) >= 35.0

                            if check_spatial(pts1_in) and check_spatial(pts2_in):
                                verified = True

        latencies.append((time.time() - t0) * 1000.0)
        stats[lbl]["total"] += 1
        stats[lbl]["inliers"].append(n_inliers)
        stats[lbl]["ratios"].append(ratio)
        if verified:
            stats[lbl]["verified"] += 1

    return stats, float(np.mean(latencies))


def main():
    print("=================================================================")
    print("   SO SÁNH TRỰC TIẾP: ORB vs SIFT (TRÊN 217 CẶP DATASET 1 & 2)  ")
    print("=================================================================")
    dets_ref = load_detections_map(JSON_REF)
    dets_qry = load_detections_map(JSON_QRY)

    with open(ALL_CSV, "r", encoding="utf-8") as f:
        all_pairs = list(csv.DictReader(f))

    for mode in ["ORB", "SIFT"]:
        stats, avg_lat = eval_matcher(all_pairs, dets_ref, dets_qry, mode=mode)
        print(f"\n--- KẾT QUẢ {mode} + DYNAMIC MASK (Latency TB: {avg_lat:.1f} ms/cặp) ---")
        for lbl in ["POSITIVE_EASY", "POSITIVE_HARD", "NEGATIVE_EASY"]:
            s = stats[lbl]
            mean_in = float(np.mean(s["inliers"])) if s["inliers"] else 0.0
            max_in = int(np.max(s["inliers"])) if s["inliers"] else 0
            mean_rat = float(np.mean(s["ratios"])) if s["ratios"] else 0.0
            ver = s["verified"]
            tot = s["total"]
            print(
                f"  {lbl:<15}: Inliers TB = {mean_in:5.1f} (Max: {max_in:3d}) | "
                f"Ratio = {mean_rat:5.1f}% | Pass Cascading Gates = {ver:3d}/{tot:3d} ({ver/tot*100:5.1f}%)"
            )


if __name__ == "__main__":
    main()
