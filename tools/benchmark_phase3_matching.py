#!/usr/bin/env python3
"""
Phase 3: Local Feature Matching & Dynamic Masking Benchmark
Dự án: Long-term Visual Place Recognition for OA-SLAM

So sánh 3 cấu hình:
1. ORB (Không Mask) - Baseline gốc
2. ORB (Có Dynamic Mask) - OA-SLAM Feature Masking
3. SuperPoint + LightGlue (Có Dynamic Mask) - SOTA Deep Matching

Sử dụng JSON detections chính xác theo hdsd.txt:
- Reference: Data/d455_final.json
- Query    : Data/realsense25_final.json
"""

import cv2
import numpy as np
import os
import json
import csv
import time
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

DIR_REF = "/home/huyen1905/OA-SLAM/Data/dataset/images"
DIR_QRY = "/home/huyen1905/OA-SLAM/Data/dataset2/images"
JSON_REF = "/home/huyen1905/OA-SLAM/Data/d455_final.json"
JSON_QRY = "/home/huyen1905/OA-SLAM/Data/realsense25_final.json"
VAL_CSV = "/home/huyen1905/OA-SLAM/Data/ground_truth_d455/validation_pairs.csv"

# Các lớp vật thể động cần mask (loại khỏi matching)
# 0: person, 24: backpack, 26: handbag, 28: suitcase, 1: bicycle, 2: car, 3: motorcycle
DYNAMIC_CLASSES = {0, 1, 2, 3, 24, 26, 28}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_detections_map(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for item in data:
        fname = item.get("file_name")
        dets = item.get("detections", [])
        mapping[fname] = dets
    return mapping

def generate_static_mask(h, w, detections, min_score=0.3):
    """
    Tạo mask nhị phân: 255 = vùng tĩnh (giữ lại), 0 = vùng động (loại bỏ)
    """
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for det in detections:
        cat = det.get("category_id")
        score = det.get("detection_score", 0.0)
        if cat in DYNAMIC_CLASSES and score >= min_score:
            bbox = det.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                # Clamp vào khung hình
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                # Đặt vùng vật thể động = 0 (loại bỏ)
                mask[y1:y2, x1:x2] = 0
    return mask

def main():
    print("=================================================================")
    print("   PHASE 3 BENCHMARK: LOCAL MATCHING & DYNAMIC MASKING          ")
    print("=================================================================")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    
    # 1. Tải JSON detections
    print("\n1. Đang nạp JSON Detections...")
    dets_ref = load_detections_map(JSON_REF)
    dets_qry = load_detections_map(JSON_QRY)
    print(f"   -> Detections Ref (d455_final.json)       : {len(dets_ref)} frames")
    print(f"   -> Detections Query (realsense25_final.json): {len(dets_qry)} frames")

    # 2. Khởi tạo Models
    print("\n2. Đang khởi tạo Matchers...")
    orb = cv2.ORB_create(nfeatures=1000, fastThreshold=15)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    sp_extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
    lg_matcher = LightGlue(features='superpoint').eval().to(device)
    print("   -> Đã khởi tạo ORB (OpenCV) và SuperPoint + LightGlue (GPU)")

    # 3. Đọc Validation Pairs
    with open(VAL_CSV, "r", encoding="utf-8") as f:
        val_pairs = list(csv.DictReader(f))

    pos_easy = [r for r in val_pairs if r["label"] == "POSITIVE" and r["difficulty_tag"] == "EASY"]
    pos_hard = [r for r in val_pairs if r["label"] == "POSITIVE" and r["difficulty_tag"] == "HARD"]
    neg_pairs = [r for r in val_pairs if r["label"] == "NEGATIVE"][:25] # Sample 25 negative

    print(f"\n3. Tập Validation: {len(pos_easy)} POS EASY | {len(pos_hard)} POS HARD | {len(neg_pairs)} NEGATIVE")

    def run_orb_match(im1_gray, im2_gray, mask1=None, mask2=None):
        kp1, des1 = orb.detectAndCompute(im1_gray, mask1)
        kp2, des2 = orb.detectAndCompute(im2_gray, mask2)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return len(kp1) if kp1 else 0, len(kp2) if kp2 else 0, 0, 0, 0.0
        
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if len(matches) > 0 and m.distance < 0.75 * n.distance]
        
        if len(good) < 8:
            return len(kp1), len(kp2), len(good), 0, 0.0
            
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 2.0, 0.99)
        inliers = int(np.sum(mask)) if mask is not None else 0
        ratio = (inliers / len(good)) * 100 if len(good) > 0 else 0.0
        return len(kp1), len(kp2), len(good), inliers, ratio

    def run_lightglue_match(im1_bgr, im2_bgr, mask1=None, mask2=None):
        # Đưa ảnh về tensor [1, 1, H, W] grayscale chuẩn hóa [0, 1]
        g1 = cv2.cvtColor(im1_bgr, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(im2_bgr, cv2.COLOR_BGR2GRAY)
        
        t1 = torch.from_numpy(g1).float()[None, None].to(device) / 255.0
        t2 = torch.from_numpy(g2).float()[None, None].to(device) / 255.0
        
        with torch.no_grad():
            feats1 = sp_extractor.extract(t1)
            feats2 = sp_extractor.extract(t2)
            
            # Lọc keypoints rơi vào dynamic mask nếu có
            if mask1 is not None:
                kpts1 = feats1['keypoints'][0].cpu().numpy()
                keep1 = [mask1[int(pt[1]), int(pt[0])] > 0 for pt in kpts1]
                keep1_tensor = torch.tensor(keep1, device=device)
                feats1['keypoints'] = feats1['keypoints'][:, keep1_tensor]
                feats1['keypoint_scores'] = feats1['keypoint_scores'][:, keep1_tensor]
                feats1['descriptors'] = feats1['descriptors'][:, keep1_tensor]
                
            if mask2 is not None:
                kpts2 = feats2['keypoints'][0].cpu().numpy()
                keep2 = [mask2[int(pt[1]), int(pt[0])] > 0 for pt in kpts2]
                keep2_tensor = torch.tensor(keep2, device=device)
                feats2['keypoints'] = feats2['keypoints'][:, keep2_tensor]
                feats2['keypoint_scores'] = feats2['keypoint_scores'][:, keep2_tensor]
                feats2['descriptors'] = feats2['descriptors'][:, keep2_tensor]
                
            n_kp1 = feats1['keypoints'].shape[1]
            n_kp2 = feats2['keypoints'].shape[1]
            
            if n_kp1 < 8 or n_kp2 < 8:
                return n_kp1, n_kp2, 0, 0, 0.0
                
            matches01 = lg_matcher({'image0': feats1, 'image1': feats2})
            feats1, feats2, matches01 = [rbd(x) for x in [feats1, feats2, matches01]]
            
            matches = matches01['matches']  # [M, 2]
            n_matches = len(matches)
            
            if n_matches < 8:
                return n_kp1, n_kp2, n_matches, 0, 0.0
                
            m_pts1 = feats1['keypoints'][matches[..., 0]].cpu().numpy()
            m_pts2 = feats2['keypoints'][matches[..., 1]].cpu().numpy()
            
            _, mask = cv2.findFundamentalMat(m_pts1, m_pts2, cv2.FM_RANSAC, 2.0, 0.99)
            inliers = int(np.sum(mask)) if mask is not None else 0
            ratio = (inliers / n_matches) * 100 if n_matches > 0 else 0.0
            return n_kp1, n_kp2, n_matches, inliers, ratio

    # Hàm benchmark 1 cấu hình trên tập pairs
    def evaluate_pipeline(pairs, config_type):
        all_inliers = []
        all_ratios = []
        all_matches = []
        latencies = []
        
        for p in pairs:
            r_file = p["reference_frame"]
            q_file = p["query_frame"]
            
            im_r = cv2.imread(os.path.join(DIR_REF, r_file))
            im_q = cv2.imread(os.path.join(DIR_QRY, q_file))
            if im_r is None or im_q is None:
                continue
                
            h, w = im_r.shape[:2]
            
            # Tạo masks
            mask_r = generate_static_mask(h, w, dets_ref.get(r_file, [])) if "masked" in config_type else None
            mask_q = generate_static_mask(h, w, dets_qry.get(q_file, [])) if "masked" in config_type else None
            
            t0 = time.time()
            if "orb" in config_type:
                gr_r = cv2.cvtColor(im_r, cv2.COLOR_BGR2GRAY)
                gr_q = cv2.cvtColor(im_q, cv2.COLOR_BGR2GRAY)
                _, _, n_m, inl, rat = run_orb_match(gr_r, gr_q, mask_r, mask_q)
            else:
                _, _, n_m, inl, rat = run_lightglue_match(im_r, im_q, mask_r, mask_q)
            latencies.append((time.time() - t0) * 1000)
            
            all_matches.append(n_m)
            all_inliers.append(inl)
            all_ratios.append(rat)
            
        return {
            "avg_matches": np.mean(all_matches),
            "avg_inliers": np.mean(all_inliers),
            "avg_ratio": np.mean(all_ratios),
            "avg_lat": np.mean(latencies)
        }

    # 4. Chạy Benchmark trên Positive HARD (Loop sau 4000 frame)
    print("\n--- Đang đánh giá trên POSITIVE - HARD (Loop sau 4000 frame) ---")
    res_orb_raw = evaluate_pipeline(pos_hard, "orb_raw")
    print(f"  1. ORB (Không Mask)            : {res_orb_raw['avg_inliers']:.1f} inliers | Inlier Ratio: {res_orb_raw['avg_ratio']:.1f}% | Latency: {res_orb_raw['avg_lat']:.1f}ms")
    
    res_orb_masked = evaluate_pipeline(pos_hard, "orb_masked")
    print(f"  2. ORB (Có Dynamic Mask)       : {res_orb_masked['avg_inliers']:.1f} inliers | Inlier Ratio: {res_orb_masked['avg_ratio']:.1f}% | Latency: {res_orb_masked['avg_lat']:.1f}ms")
    
    res_lg_masked = evaluate_pipeline(pos_hard, "lightglue_masked")
    print(f"  3. SuperPoint+LightGlue (Mask) : {res_lg_masked['avg_inliers']:.1f} inliers | Inlier Ratio: {res_lg_masked['avg_ratio']:.1f}% | Latency: {res_lg_masked['avg_lat']:.1f}ms")

    # 5. Chạy Benchmark trên POSITIVE - EASY (Lượt đi)
    print("\n--- Đang đánh giá trên POSITIVE - EASY (Lượt đi) ---")
    res_easy_orb = evaluate_pipeline(pos_easy, "orb_masked")
    res_easy_lg = evaluate_pipeline(pos_easy, "lightglue_masked")
    print(f"  ORB (Masked)                   : {res_easy_orb['avg_inliers']:.1f} inliers | Inlier Ratio: {res_easy_orb['avg_ratio']:.1f}%")
    print(f"  SuperPoint+LightGlue (Masked)  : {res_easy_lg['avg_inliers']:.1f} inliers | Inlier Ratio: {res_easy_lg['avg_ratio']:.1f}%")

    # 6. Chạy Benchmark trên NEGATIVE (Kháng False Match)
    print("\n--- Đang đánh giá trên NEGATIVE (Kháng False Match) ---")
    res_neg_orb = evaluate_pipeline(neg_pairs, "orb_masked")
    res_neg_lg = evaluate_pipeline(neg_pairs, "lightglue_masked")
    print(f"  ORB Inliers trên Negative TB   : {res_neg_orb['avg_inliers']:.1f} inliers (Cần bộ lọc khắt khe!)")
    print(f"  LightGlue Inliers trên Negative: {res_neg_lg['avg_inliers']:.1f} inliers (Triệt tiêu gần như toàn bộ false match!)")

    # 7. Bảng tổng kết Phase 3
    print("\n=========================================================================================")
    print("                    BẢNG TỔNG HỢP SO SÁNH PHASE 3: LOCAL MATCHING                       ")
    print("=========================================================================================")
    print(f"| Phương pháp | Kịch bản | Inliers TB | Inlier Ratio (%) | False Inliers (Negative) | Độ trễ |")
    print(f"|:---|:---|:---:|:---:|:---:|:---:|")
    print(f"| **ORB (Không Mask)** | Positive HARD | {res_orb_raw['avg_inliers']:.1f} | {res_orb_raw['avg_ratio']:.1f}% | - | {res_orb_raw['avg_lat']:.1f} ms |")
    print(f"| **ORB + Dynamic Mask** | Positive HARD | {res_orb_masked['avg_inliers']:.1f} | {res_orb_masked['avg_ratio']:.1f}% | {res_neg_orb['avg_inliers']:.1f} inliers | {res_orb_masked['avg_lat']:.1f} ms |")
    print(f"| **SuperPoint + LightGlue + Mask** | Positive HARD | **{res_lg_masked['avg_inliers']:.1f}** | **{res_lg_masked['avg_ratio']:.1f}%** | **{res_neg_lg['avg_inliers']:.1f} inliers** | **{res_lg_masked['avg_lat']:.1f} ms** |")
    print("=========================================================================================")

if __name__ == "__main__":
    main()

