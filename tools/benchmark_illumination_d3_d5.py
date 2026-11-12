#!/usr/bin/env python3
"""
Benchmark So sánh Độ sáng (Illumination Change: Sáng - Tối / Bật đèn - Tắt đèn trần)
1. Thực nghiệm trên 2 bộ dữ liệu Sảnh tòa nhà thực tế:
   - Data/dataset3 (675 frames, realsense35_final.json): Sảnh bật đèn trần sáng
   - Data/dataset5 (1413 frames, realsense55_final.json): Cùng sảnh nhưng tắt đèn trần (tối/ngược sáng cửa kính)
2. Thực nghiệm Stress-Test Biến thiên Độ sáng Cực đoan (Extreme Gamma/Exposure Shift):
   - Rất tối / Tắt đèn (gamma = 0.35)
   - Tối vừa / Chiều muộn (gamma = 0.55)
   - Độ sáng gốc (gamma = 1.00)
   - Chói sáng (gamma = 1.80)
   - Ngược sáng gắt (gamma = 2.50)

So sánh 3 phương pháp trích xuất & đối sánh đặc trưng cục bộ đi qua Cổng kiểm chứng hình học Cascading Gates:
  1. ORB + Dynamic Mask (Hamming KNN + Ratio 0.75)
  2. SIFT + Dynamic Mask (L2 KNN + Lowe's Ratio 0.75 + Mutual Nearest Neighbor)
  3. CLAHE-SIFT + Dynamic Mask (Chuẩn hóa tương phản thích ứng cục bộ + SIFT + Mutual NN — cấu hình được tích hợp vào C++ OA-SLAM)
"""

import os
import csv
import json
import time
import cv2
import numpy as np

DIR_D3 = "/home/huyen1905/OA-SLAM/Data/dataset3/images"
DIR_D5 = "/home/huyen1905/OA-SLAM/Data/dataset5/images"
JSON_D3 = "/home/huyen1905/OA-SLAM/Data/realsense35_final.json"
JSON_D5 = "/home/huyen1905/OA-SLAM/Data/realsense55_final.json"

DIR_REF = "/home/huyen1905/OA-SLAM/Data/dataset/images"
DIR_QRY = "/home/huyen1905/OA-SLAM/Data/dataset2/images"
JSON_REF = "/home/huyen1905/OA-SLAM/Data/d455_final.json"
JSON_QRY = "/home/huyen1905/OA-SLAM/Data/realsense25_final.json"
ALL_CSV = "/home/huyen1905/OA-SLAM/Data/ground_truth_d455/all_pairs.csv"
OUT_REPORT = "/home/huyen1905/OA-SLAM/Data/ground_truth_d455/illumination_d3_d5_report.json"

K_D455 = np.array([
    [393.7687, 0.0, 315.1953],
    [0.0, 393.7962, 239.3341],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DYNAMIC_CLASSES = {0, 1, 2, 3, 24, 26, 28}


def load_det_index(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx = {}
    for entry in data:
        fname = os.path.basename(entry.get("file_name", ""))
        idx[fname] = entry.get("detections", [])
    return idx


def generate_static_mask(h, w, detections, min_score=0.3):
    mask = np.full((h, w), 255, dtype=np.uint8)
    for det in detections:
        if det.get("category_id") in DYNAMIC_CLASSES and det.get("detection_score", 0.0) >= min_score:
            bbox = det.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                mask[max(0, y1 - 8):min(h, y2 + 8), max(0, x1 - 8):min(w, x2 + 8)] = 0
    return mask


def apply_gamma(img, gamma):
    if abs(gamma - 1.0) < 1e-3:
        return img
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def check_spatial(pts, h, w):
    grid = np.zeros((3, 3), dtype=np.int32)
    for x, y in pts:
        grid[min(2, int(y / (h / 3))), min(2, int(x / (w / 3)))] += 1
    occ = int(np.sum(grid >= 2))
    hull = cv2.contourArea(cv2.convexHull(pts)) / (w * h)
    return occ >= 4 and hull >= 0.12 and np.std(pts[:, 0]) >= 50.0 and np.std(pts[:, 1]) >= 30.0


def match_and_verify(im1, im2, m1, m2, mode="SIFT"):
    h, w = im1.shape[:2]
    t0 = time.time()

    if mode == "ORB":
        extractor = cv2.ORB_create(nfeatures=2000, fastThreshold=8)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        kp1, des1 = extractor.detectAndCompute(im1, m1)
        kp2, des2 = extractor.detectAndCompute(im2, m2)
        if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
            return 0, 0.0, False, (time.time() - t0) * 1000.0
        knn = bf.knnMatch(des1, des2, k=2)
        good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]) if good else np.empty((0, 2))
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]) if good else np.empty((0, 2))

    elif mode in ("SIFT", "CLAHE_SIFT"):
        if mode == "CLAHE_SIFT":
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            im1_proc = clahe.apply(im1)
            im2_proc = clahe.apply(im2)
        else:
            im1_proc = im1
            im2_proc = im2
        extractor = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.03)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        kp1, des1 = extractor.detectAndCompute(im1_proc, m1)
        kp2, des2 = extractor.detectAndCompute(im2_proc, m2)
        if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
            return 0, 0.0, False, (time.time() - t0) * 1000.0
        knn12 = bf.knnMatch(des1, des2, k=2)
        knn21 = bf.knnMatch(des2, des1, k=2)
        g12 = {p[0].queryIdx: p[0].trainIdx for p in knn12 if len(p) == 2 and p[0].distance < 0.78 * p[1].distance}
        g21 = {p[0].queryIdx: p[0].trainIdx for p in knn21 if len(p) == 2 and p[0].distance < 0.78 * p[1].distance}
        mutual = [(i, j) for i, j in g12.items() if g21.get(j, -1) == i]
        pts1 = np.float32([kp1[i].pt for i, _ in mutual]) if mutual else np.empty((0, 2))
        pts2 = np.float32([kp2[j].pt for _, j in mutual]) if mutual else np.empty((0, 2))
    else:
        raise ValueError(mode)

    if len(pts1) < 12:
        return 0, 0.0, False, (time.time() - t0) * 1000.0

    E, mask_E = cv2.findEssentialMat(pts1, pts2, K_D455, method=cv2.RANSAC, prob=0.999, threshold=2.0)
    if E is None or mask_E is None:
        return 0, 0.0, False, (time.time() - t0) * 1000.0

    idx_in = np.where(mask_E.ravel() == 1)[0]
    n_inliers = len(idx_in)
    ratio = (n_inliers / len(pts1)) * 100.0 if len(pts1) > 0 else 0.0
    verified = False
    if n_inliers >= 25 and (n_inliers / len(pts1)) >= 0.35:
        pts1_in = pts1[idx_in]
        pts2_in = pts2[idx_in]
        num_pos, _, _, _ = cv2.recoverPose(E, pts1_in, pts2_in, K_D455)
        if (num_pos / max(1, n_inliers)) >= 0.50:
            if check_spatial(pts1_in, h, w) and check_spatial(pts2_in, h, w):
                verified = True

    return n_inliers, ratio, verified, (time.time() - t0) * 1000.0


def main():
    print("==========================================================================")
    print("1. KHÁM PHÁ & ĐỐI SÁNH CẶP SẢNH SÁNG - TỐI: DATASET3 (BẬT ĐÈN) vs DATASET5 (TẮT ĐÈN)")
    print("==========================================================================")
    dets_d3 = load_det_index(JSON_D3)
    dets_d5 = load_det_index(JSON_D5)
    imgs_d3 = sorted([f for f in os.listdir(DIR_D3) if f.endswith(".png")])
    imgs_d5 = sorted([f for f in os.listdir(DIR_D5) if f.endswith(".png")])

    # Quét tìm các cặp cùng vị trí trong sảnh giữa dataset3 và dataset5
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    sift_scan = cv2.SIFT_create(nfeatures=800, contrastThreshold=0.03)
    bf_l2 = cv2.BFMatcher(cv2.NORM_L2)

    d3_cache = []
    for f3 in imgs_d3[::15]:
        im3 = cv2.imread(os.path.join(DIR_D3, f3), cv2.IMREAD_GRAYSCALE)
        if im3 is None:
            continue
        m3 = generate_static_mask(im3.shape[0], im3.shape[1], dets_d3.get(f3, []))
        kp3, des3 = sift_scan.detectAndCompute(clahe.apply(im3), m3)
        if des3 is not None and len(kp3) >= 20:
            d3_cache.append((f3, im3, m3, kp3, des3, float(im3.mean())))

    d5_cache = []
    for f5 in imgs_d5[::20]:
        im5 = cv2.imread(os.path.join(DIR_D5, f5), cv2.IMREAD_GRAYSCALE)
        if im5 is None:
            continue
        m5 = generate_static_mask(im5.shape[0], im5.shape[1], dets_d5.get(f5, []))
        kp5, des5 = sift_scan.detectAndCompute(clahe.apply(im5), m5)
        if des5 is not None and len(kp5) >= 20:
            d5_cache.append((f5, im5, m5, kp5, des5, float(im5.mean())))

    pos_d3_d5 = []
    neg_d3_d5 = []
    for f3, im3, m3, kp3, des3, mean3 in d3_cache:
        best_match = None
        best_inl = 0
        for f5, im5, m5, kp5, des5, mean5 in d5_cache:
            knn = bf_l2.knnMatch(des3, des5, k=2)
            good = [m for p in knn if len(p) == 2 for m, n in [p] if m.distance < 0.78 * n.distance]
            if len(good) >= 22:
                pts1 = np.float32([kp3[m.queryIdx].pt for m in good])
                pts2 = np.float32([kp5[m.trainIdx].pt for m in good])
                E, mask = cv2.findEssentialMat(pts1, pts2, K_D455, method=cv2.RANSAC, prob=0.999, threshold=2.0)
                if mask is not None:
                    inl = int(mask.ravel().sum())
                    if inl >= 20 and inl > best_inl:
                        best_inl = inl
                        best_match = (f3, f5, mean3, mean5, inl)
            elif 6 <= len(good) <= 12 and len(neg_d3_d5) < 40:
                idx3 = int(os.path.splitext(f3)[0])
                idx5 = int(os.path.splitext(f5)[0])
                if abs(idx3 - idx5) > 300:
                    neg_d3_d5.append((f3, f5, mean3, mean5))
        if best_match is not None:
            pos_d3_d5.append(best_match)

    print(f"-> Tìm thấy {len(pos_d3_d5)} cặp trùng địa điểm giữa dataset3 (Sảnh Bật Đèn) và dataset5 (Sảnh Tắt Đèn Trần)")
    for f3, f5, mean3, mean5, inl in pos_d3_d5[:8]:
        print(f"   [Cặp Sảnh] dataset3/{f3} (Sáng={mean3:.1f}) <-> dataset5/{f5} (Sáng={mean5:.1f}) | ΔL={abs(mean3-mean5):.1f}")

    lobby_report = {}
    for mode in ["ORB", "SIFT", "CLAHE_SIFT"]:
        pos_in, pos_rat, pos_ver, lat = [], [], 0, []
        neg_fp = 0
        for f3, f5, _, _, _ in pos_d3_d5:
            im3 = cv2.imread(os.path.join(DIR_D3, f3), cv2.IMREAD_GRAYSCALE)
            im5 = cv2.imread(os.path.join(DIR_D5, f5), cv2.IMREAD_GRAYSCALE)
            m3 = generate_static_mask(im3.shape[0], im3.shape[1], dets_d3.get(f3, []))
            m5 = generate_static_mask(im5.shape[0], im5.shape[1], dets_d5.get(f5, []))
            inl, rat, ver, dt = match_and_verify(im3, im5, m3, m5, mode=mode)
            pos_in.append(inl)
            pos_rat.append(rat)
            pos_ver += int(ver)
            lat.append(dt)
        for f3, f5, _, _ in neg_d3_d5:
            im3 = cv2.imread(os.path.join(DIR_D3, f3), cv2.IMREAD_GRAYSCALE)
            im5 = cv2.imread(os.path.join(DIR_D5, f5), cv2.IMREAD_GRAYSCALE)
            m3 = generate_static_mask(im3.shape[0], im3.shape[1], dets_d3.get(f3, []))
            m5 = generate_static_mask(im5.shape[0], im5.shape[1], dets_d5.get(f5, []))
            _, _, ver, _ = match_and_verify(im3, im5, m3, m5, mode=mode)
            neg_fp += int(ver)
        n_p = max(1, len(pos_d3_d5))
        n_n = max(1, len(neg_d3_d5))
        lobby_report[mode] = {
            "avg_inliers": round(float(np.mean(pos_in)), 1),
            "avg_ratio_pct": round(float(np.mean(pos_rat)), 1),
            "recall_pct": round(100.0 * pos_ver / n_p, 1),
            "passed": pos_ver,
            "total_pos": len(pos_d3_d5),
            "fpr_pct": round(100.0 * neg_fp / n_n, 2),
            "false_positives": neg_fp,
            "total_neg": len(neg_d3_d5),
            "latency_ms": round(float(np.mean(lat)), 1),
        }
        print(
            f"  [{mode:<10}] Inliers TB: {lobby_report[mode]['avg_inliers']:5.1f} | "
            f"Ratio: {lobby_report[mode]['avg_ratio_pct']:5.1f}% | "
            f"Pass Gates: {pos_ver}/{len(pos_d3_d5)} ({lobby_report[mode]['recall_pct']:5.1f}%) | "
            f"FPR: {neg_fp}/{len(neg_d3_d5)} ({lobby_report[mode]['fpr_pct']:4.1f}%) | "
            f"{lobby_report[mode]['latency_ms']:5.1f} ms"
        )

    print("\n==========================================================================")
    print("2. STRESS-TEST ĐỘ SÁNG (SÁNG vs TỐI CỰC ĐOAN) TRÊN 53 CẶP POSITIVE (DATASET 1 & 2)")
    print("==========================================================================")
    dets_ref = load_det_index(JSON_REF)
    dets_qry = load_det_index(JSON_QRY)
    with open(ALL_CSV, "r", encoding="utf-8") as f:
        pos_pairs = [r for r in csv.DictReader(f) if r["label"] == "POSITIVE"]

    gamma_levels = [
        ("Rất tối / Đêm (gamma=0.35)", 0.35),
        ("Tối vừa / Tắt đèn (gamma=0.55)", 0.55),
        ("Độ sáng gốc (gamma=1.00)", 1.00),
        ("Chói sáng (gamma=1.80)", 1.80),
        ("Ngược sáng gắt (gamma=2.50)", 2.50),
    ]
    stress_report = {}
    for g_label, g_val in gamma_levels:
        stress_report[g_label] = {}
        print(f"\n--- Điều kiện ánh sáng Query: {g_label} ---")
        for mode in ["ORB", "SIFT", "CLAHE_SIFT"]:
            inls, rats, vers = [], [], 0
            for p in pos_pairs:
                r_file = p["reference_frame"]
                q_file = p["query_frame"]
                im1 = cv2.imread(os.path.join(DIR_REF, r_file), cv2.IMREAD_GRAYSCALE)
                im2 = apply_gamma(cv2.imread(os.path.join(DIR_QRY, q_file), cv2.IMREAD_GRAYSCALE), g_val)
                m1 = generate_static_mask(im1.shape[0], im1.shape[1], dets_ref.get(r_file, []))
                m2 = generate_static_mask(im2.shape[0], im2.shape[1], dets_qry.get(q_file, []))
                inl, rat, ver, _ = match_and_verify(im1, im2, m1, m2, mode=mode)
                inls.append(inl)
                rats.append(rat)
                vers += int(ver)
            rec = 100.0 * vers / len(pos_pairs)
            stress_report[g_label][mode] = {
                "avg_inliers": round(float(np.mean(inls)), 1),
                "avg_ratio_pct": round(float(np.mean(rats)), 1),
                "recall_pct": round(rec, 1),
                "passed": vers,
                "total": len(pos_pairs),
            }
            print(
                f"  [{mode:<10}] Inliers TB = {np.mean(inls):5.1f} | "
                f"Ratio = {np.mean(rats):5.1f}% | Pass Gates = {vers:2d}/{len(pos_pairs)} ({rec:5.1f}%)"
            )

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump({"lobby_dataset3_vs_dataset5": lobby_report, "illumination_stress_test": stress_report}, f, indent=2)
    print(f"\nĐã lưu báo cáo tại: {OUT_REPORT}")


if __name__ == "__main__":
    main()
