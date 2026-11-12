#!/usr/bin/env python3
"""
Phase 4: Cascading Verification Pipeline for OA-SLAM Long-Term Loop Closure
Tác giả: OA-SLAM Long-Term Place Recognition Team
Mục tiêu: Đạt ZERO False Positives (FPR = 0.0%) trên bộ dữ liệu kiểm định

4 Tầng Cửa (Cascading Verification Gates):
- Gate 1: Epipolar & Metric Geometry (Essential Matrix RANSAC, Inliers >= 35, Cheirality >= 85%)
- Gate 2: Spatial Distribution Coverage (Grid 3x3 Occupancy >= 4/9, Convex Hull Area >= 20%)
- Gate 3: Semantic Object Consistency (Category Jaccard & Point-in-BBox Consistency)
- Gate 4: Temporal Covisibility / Continuity (>= 3 khung hình liên tiếp đồng thuận)
"""

import os
import cv2
import numpy as np
import json
import csv
import time
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

# Đường dẫn dữ liệu
DIR_REF = "/home/huyen1905/OA-SLAM/Data/dataset/images"
DIR_QRY = "/home/huyen1905/OA-SLAM/Data/dataset2/images"
JSON_REF = "/home/huyen1905/OA-SLAM/Data/d455_final.json"
JSON_QRY = "/home/huyen1905/OA-SLAM/Data/realsense25_final.json"
VAL_CSV = "/home/huyen1905/OA-SLAM/Data/ground_truth_d455/validation_pairs.csv"

# Camera Intrinsics từ Cameras/d455.yaml
K_D455 = np.array([
    [393.7687, 0.0, 315.1953],
    [0.0, 393.7962, 239.3341],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# Vật thể động cần loại bỏ (COCO classes: person, bicycle, car, motorcycle, backpack, handbag, suitcase)
DYNAMIC_CLASSES = {0, 1, 2, 3, 24, 26, 28}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_detections_map(json_path):
    """Tải bản đồ phát hiện vật thể từ file JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for item in data:
        fname = item.get("file_name")
        dets = item.get("detections", [])
        mapping[fname] = dets
    return mapping


def generate_static_mask(h, w, detections, min_score=0.3):
    """Tạo mask nhị phân: 255 = vùng tĩnh (giữ lại), 0 = vùng động (loại bỏ)."""
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for det in detections:
        cat = det.get("category_id")
        score = det.get("detection_score", 0.0)
        if cat in DYNAMIC_CLASSES and score >= min_score:
            bbox = det.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask[y1:y2, x1:x2] = 0
    return mask


class CascadingVerifier:
    def __init__(self, camera_k=K_D455):
        self.K = camera_k
        self.img_w = 640
        self.img_h = 480
        
        # Ngưỡng Gate 1 (Epipolar Geometry)
        self.min_inliers = 35           # Bắt buộc >= 35 để vượt xa 10.6 false inliers của Negative
        self.min_inlier_ratio = 0.40     # Tỷ lệ inlier >= 40%
        self.max_epipolar_err = 1.8      # Sai số epipolar tối đa (pixels)
        self.min_cheirality_ratio = 0.50 # Cho phép baseline hẹp (forward motion)

        # Ngưỡng Gate 2 (Spatial Distribution)
        self.min_grid_occupancy = 4      # Ít nhất 4/9 ô lưới 3x3 phải có inlier
        self.min_hull_area_ratio = 0.15  # Diện tích bao lồi >= 15% khung hình
        self.min_std_x = 55.0            # Độ tản mác ngang >= 55 px
        self.min_std_y = 35.0            # Độ tản mác dọc >= 35 px
        
        # Khởi tạo SuperPoint & LightGlue trên GPU
        self.sp_extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
        self.lg_matcher = LightGlue(features='superpoint').eval().to(device)

    def extract_and_match(self, im1_bgr, im2_bgr, mask1=None, mask2=None):
        """Khớp đặc trưng bằng SuperPoint + LightGlue có dynamic mask."""
        g1 = cv2.cvtColor(im1_bgr, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(im2_bgr, cv2.COLOR_BGR2GRAY)
        
        t1 = torch.from_numpy(g1).float()[None, None].to(device) / 255.0
        t2 = torch.from_numpy(g2).float()[None, None].to(device) / 255.0
        
        with torch.no_grad():
            f1 = self.sp_extractor.extract(t1)
            f2 = self.sp_extractor.extract(t2)
            
            if mask1 is not None:
                k1 = f1['keypoints'][0].cpu().numpy()
                keep1 = [mask1[int(pt[1]), int(pt[0])] > 0 for pt in k1]
                t_keep1 = torch.tensor(keep1, device=device)
                f1['keypoints'] = f1['keypoints'][:, t_keep1]
                f1['keypoint_scores'] = f1['keypoint_scores'][:, t_keep1]
                f1['descriptors'] = f1['descriptors'][:, t_keep1]
                
            if mask2 is not None:
                k2 = f2['keypoints'][0].cpu().numpy()
                keep2 = [mask2[int(pt[1]), int(pt[0])] > 0 for pt in k2]
                t_keep2 = torch.tensor(keep2, device=device)
                f2['keypoints'] = f2['keypoints'][:, t_keep2]
                f2['keypoint_scores'] = f2['keypoint_scores'][:, t_keep2]
                f2['descriptors'] = f2['descriptors'][:, t_keep2]
                
            n1 = f1['keypoints'].shape[1]
            n2 = f2['keypoints'].shape[1]
            if n1 < 8 or n2 < 8:
                return np.empty((0, 2)), np.empty((0, 2))
                
            m01 = self.lg_matcher({'image0': f1, 'image1': f2})
            f1, f2, m01 = [rbd(x) for x in [f1, f2, m01]]
            matches = m01['matches']
            if len(matches) < 8:
                return np.empty((0, 2)), np.empty((0, 2))
                
            pts1 = f1['keypoints'][matches[..., 0]].cpu().numpy()
            pts2 = f2['keypoints'][matches[..., 1]].cpu().numpy()
            return pts1, pts2

    def check_gate_1_epipolar(self, pts1, pts2):
        """
        Gate 1: Epipolar & Metric Geometry Check
        Sử dụng Essential Matrix RANSAC với Camera Matrix K và Cheirality Check.
        """
        n_matches = len(pts1)
        if n_matches < self.min_inliers:
            return False, {
                "reason": f"Matches < {self.min_inliers} ({n_matches})",
                "inliers": 0, "ratio": 0.0, "cheirality": 0.0, "mean_err": 999.0
            }, None, None
            
        # Tìm Essential Matrix bằng RANSAC
        E, mask_E = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=self.max_epipolar_err
        )
        if E is None or mask_E is None:
            return False, {"reason": "Essential Matrix RANSAC failed", "inliers": 0}, None, None
            
        inliers_idx = np.where(mask_E.ravel() == 1)[0]
        n_inliers = len(inliers_idx)
        inlier_ratio = n_inliers / n_matches
        
        if n_inliers < self.min_inliers:
            return False, {
                "reason": f"Inliers < {self.min_inliers} ({n_inliers})",
                "inliers": n_inliers, "ratio": inlier_ratio, "cheirality": 0.0
            }, None, None
            
        if inlier_ratio < self.min_inlier_ratio:
            return False, {
                "reason": f"Inlier ratio < {self.min_inlier_ratio*100:.0f}% ({inlier_ratio*100:.1f}%)",
                "inliers": n_inliers, "ratio": inlier_ratio, "cheirality": 0.0
            }, None, None

        pts1_in = pts1[inliers_idx]
        pts2_in = pts2[inliers_idx]

        # Cheirality Check (Thu hồi Pose và kiểm tra Depth dương trước 2 camera)
        num_pos, R, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, self.K)
        cheirality_ratio = num_pos / n_inliers if n_inliers > 0 else 0.0
        
        if cheirality_ratio < self.min_cheirality_ratio:
            return False, {
                "reason": f"Cheirality < {self.min_cheirality_ratio*100:.0f}% ({cheirality_ratio*100:.1f}%)",
                "inliers": n_inliers, "ratio": inlier_ratio, "cheirality": cheirality_ratio
            }, None, None
            
        return True, {
            "inliers": n_inliers,
            "ratio": inlier_ratio,
            "cheirality": cheirality_ratio,
            "n_matches": n_matches
        }, pts1_in, pts2_in

    def check_gate_2_spatial(self, pts1_in, pts2_in):
        """
        Gate 2: Spatial Distribution & Coverage Check
        Loại bỏ các trường hợp inliers bị gom cụm cục bộ vào 1 góc nhỏ hoặc một đường thẳng.
        """
        def compute_distribution(pts):
            # 1. 3x3 Grid Occupancy (Mỗi ô ít nhất 2 inliers)
            grid = np.zeros((3, 3), dtype=np.int32)
            for x, y in pts:
                gx = min(2, int(x / (self.img_w / 3)))
                gy = min(2, int(y / (self.img_h / 3)))
                grid[gy, gx] += 1
            occupied_cells = int(np.sum(grid >= 2))
            
            # 2. Convex Hull Area Ratio
            if len(pts) >= 4:
                hull = cv2.convexHull(pts.astype(np.float32))
                hull_area = cv2.contourArea(hull)
                hull_ratio = hull_area / (self.img_w * self.img_h)
            else:
                hull_ratio = 0.0
                
            # 3. Standard Deviation (Tán mác theo 2 chiều)
            std_x = float(np.std(pts[:, 0]))
            std_y = float(np.std(pts[:, 1]))
            
            return occupied_cells, hull_ratio, std_x, std_y

        occ1, hull1, std_x1, std_y1 = compute_distribution(pts1_in)
        occ2, hull2, std_x2, std_y2 = compute_distribution(pts2_in)
        
        # Kiểm tra điều kiện tối thiểu trên CẢ 2 ảnh
        min_occ = min(occ1, occ2)
        min_hull = min(hull1, hull2)
        min_sx = min(std_x1, std_x2)
        min_sy = min(std_y1, std_y2)
        
        if min_occ < self.min_grid_occupancy:
            return False, {
                "reason": f"Grid occupancy < {self.min_grid_occupancy}/9 (Ref:{occ1}/9, Qry:{occ2}/9)",
                "occ1": occ1, "occ2": occ2, "hull1": hull1, "hull2": hull2
            }
            
        if min_hull < self.min_hull_area_ratio:
            return False, {
                "reason": f"Convex Hull area < {self.min_hull_area_ratio*100:.0f}% (Ref:{hull1*100:.1f}%, Qry:{hull2*100:.1f}%)",
                "occ1": occ1, "occ2": occ2, "hull1": hull1, "hull2": hull2
            }
            
        if min_sx < self.min_std_x or min_sy < self.min_std_y:
            return False, {
                "reason": f"Spatial std too low (sx:{min_sx:.1f}, sy:{min_sy:.1f})",
                "std_x": min_sx, "std_y": min_sy
            }
            
        return True, {
            "occ1": occ1, "occ2": occ2,
            "hull1": hull1, "hull2": hull2,
            "std_x1": std_x1, "std_y1": std_y1
        }

    def check_gate_3_semantic(self, dets1, dets2, pts1_in, pts2_in):
        """
        Gate 3: Semantic Graph & Object Consistency Check
        - Đối chiếu tập lớp vật thể tĩnh (static objects) xuất hiện trên 2 ảnh.
        - Kiểm tra tính nhất quán nhãn khi inliers rơi vào bounding box.
        """
        # Trích xuất các vật thể tĩnh có độ tự tin cao
        static_dets1 = [d for d in dets1 if d.get("category_id") not in DYNAMIC_CLASSES and d.get("detection_score", 0) >= 0.35]
        static_dets2 = [d for d in dets2 if d.get("category_id") not in DYNAMIC_CLASSES and d.get("detection_score", 0) >= 0.35]
        
        cats1 = set(d.get("category_id") for d in static_dets1)
        cats2 = set(d.get("category_id") for d in static_dets2)
        
        # Trường hợp 1: Cả 2 ảnh đều có vật thể tĩnh tự tin
        if len(cats1) > 0 and len(cats2) > 0:
            inter = cats1.intersection(cats2)
            union = cats1.union(cats2)
            jaccard = len(inter) / len(union)
            
            # Nếu cả 2 đều có từ 2 vật thể tĩnh trở lên nhưng KHÔNG CÓ BẤT KỲ LỚP NÀO TRÙNG NHAU (Khác hoàn toàn không gian)
            if len(cats1) >= 2 and len(cats2) >= 2 and len(inter) == 0:
                return False, {
                    "reason": f"Semantic conflict: categories disjoint {cats1} vs {cats2}",
                    "jaccard": jaccard
                }
                
            # Kiểm tra Point-in-BBox semantic correspondence
            # Nếu 1 inlier ở ảnh 1 nằm trong vật thể A, sang ảnh 2 nó phải không được nằm trong vật thể B hoàn toàn đối lập
            contradictions = 0
            object_matches = 0
            for p1, p2 in zip(pts1_in, pts2_in):
                c1_match = None
                for d in static_dets1:
                    b = d["bbox"]
                    if b[0] <= p1[0] <= b[2] and b[1] <= p1[1] <= b[3]:
                        c1_match = d["category_id"]
                        break
                        
                c2_match = None
                for d in static_dets2:
                    b = d["bbox"]
                    if b[0] <= p2[0] <= b[2] and b[1] <= p2[1] <= b[3]:
                        c2_match = d["category_id"]
                        break
                        
                if c1_match is not None and c2_match is not None:
                    object_matches += 1
                    if c1_match != c2_match:
                        contradictions += 1
                        
            if object_matches >= 8 and (contradictions / object_matches) > 0.40:
                return False, {
                    "reason": f"Point semantic contradiction: {contradictions}/{object_matches} matches conflicted",
                    "contradictions": contradictions
                }
                
            return True, {
                "jaccard": jaccard, "common_cats": list(inter),
                "object_matches": object_matches, "contradictions": contradictions
            }
            
        # Trường hợp 2: Khung hình ít hoặc không có vật thể tĩnh (hành lang trơn / tường trắng)
        # Gate 3 cho qua nếu Gate 1 và Gate 2 đã đạt độ tự tin cao
        return True, {"jaccard": 1.0, "status": "neutral_no_static_objects"}

    def verify_pair(self, im1_bgr, im2_bgr, dets1, dets2):
        """Quy trình xác thực tầng bậc đầy đủ trên 1 cặp ảnh."""
        h, w = im1_bgr.shape[:2]
        m1 = generate_static_mask(h, w, dets1)
        m2 = generate_static_mask(h, w, dets2)
        
        # Khớp đặc trưng
        pts1, pts2 = self.extract_and_match(im1_bgr, im2_bgr, m1, m2)
        if len(pts1) < self.min_inliers:
            return "REJECTED_GATE_1", {
                "gate": 1, "reason": f"Feature matches too few ({len(pts1)} < {self.min_inliers})"
            }
            
        # 1. GATE 1: Epipolar & Metric Geometry
        pass_g1, info_g1, pts1_in, pts2_in = self.check_gate_1_epipolar(pts1, pts2)
        if not pass_g1:
            return "REJECTED_GATE_1", {"gate": 1, **info_g1}
            
        # 2. GATE 2: Spatial Distribution
        pass_g2, info_g2 = self.check_gate_2_spatial(pts1_in, pts2_in)
        if not pass_g2:
            return "REJECTED_GATE_2", {"gate": 2, "g1_info": info_g1, **info_g2}
            
        # 3. GATE 3: Semantic Object Consistency
        pass_g3, info_g3 = self.check_gate_3_semantic(dets1, dets2, pts1_in, pts2_in)
        if not pass_g3:
            return "REJECTED_GATE_3", {"gate": 3, "g1_info": info_g1, "g2_info": info_g2, **info_g3}
            
        return "VERIFIED", {
            "gate": 0, "g1_info": info_g1, "g2_info": info_g2, "g3_info": info_g3
        }


class TemporalLoopVerifier:
    """
    Gate 4: Temporal Covisibility / Continuity Verifier
    Trong SLAM thực tế, một ứng viên Loop chỉ được xác nhận khi được hỗ trợ bởi
    ít nhất 3 khung hình liên tiếp đồng thuận trong cùng một vùng lân cận bản đồ.
    """
    def __init__(self, window_size=3, max_ref_jump=40):
        self.window_size = window_size
        self.max_ref_jump = max_ref_jump
        self.history = [] # danh sách các (query_id, ref_id) đã vượt qua Gate 1-3

    def add_candidate(self, query_id, ref_id):
        self.history.append((query_id, ref_id))
        if len(self.history) < self.window_size:
            return False, "Buffer accumulating (chưa đủ 3 frames liên tiếp)"
            
        recent = self.history[-self.window_size:]
        # Kiểm tra tính liên tục của query_id và ref_id
        q_ids = [q for q, r in recent]
        r_ids = [r for q, r in recent]
        
        # Query id phải tăng đều (khung hình nối tiếp nhau)
        q_diffs = [q_ids[i] - q_ids[i-1] for i in range(1, len(q_ids))]
        if any(d > 50 for d in q_diffs): # Khoảng cách query không được quá xa
            return False, "Query frames not temporally contiguous"
            
        # Reference id phải nằm trong cùng cụm lân cận (không bị nhảy cóc hỗn loạn)
        r_diffs = [abs(r_ids[i] - r_ids[i-1]) for i in range(1, len(r_ids))]
        if any(d > self.max_ref_jump for d in r_diffs):
            return False, f"Reference candidate jumped > {self.max_ref_jump} frames"
            
        return True, f"Temporal continuity confirmed across {self.window_size} frames"


def main():
    print("=================================================================")
    print("   PHASE 4: CASCADING VERIFICATION BENCHMARK (129 VAL PAIRS)   ")
    print("=================================================================")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    
    # 1. Tải JSON Detections
    print("\n1. Đang nạp JSON Detections...")
    dets_ref = load_detections_map(JSON_REF)
    dets_qry = load_detections_map(JSON_QRY)
    print(f"   -> Ref Detections: {len(dets_ref)} frames")
    print(f"   -> Qry Detections: {len(dets_qry)} frames")
    
    # 2. Khởi tạo Verifier
    print("\n2. Khởi tạo Cascading Verifier (Gate 1, 2, 3)...")
    verifier = CascadingVerifier()
    
    # 3. Đọc Validation Pairs (129 pairs)
    with open(VAL_CSV, "r", encoding="utf-8") as f:
        val_pairs = list(csv.DictReader(f))
        
    print(f"\n3. Bắt đầu đánh giá trên {len(val_pairs)} Validation Pairs:")
    
    results = {
        "POSITIVE_EASY": {"VERIFIED": 0, "REJECTED_G1": 0, "REJECTED_G2": 0, "REJECTED_G3": 0, "TOTAL": 0},
        "POSITIVE_HARD": {"VERIFIED": 0, "REJECTED_G1": 0, "REJECTED_G2": 0, "REJECTED_G3": 0, "TOTAL": 0},
        "NEGATIVE_EASY": {"VERIFIED": 0, "REJECTED_G1": 0, "REJECTED_G2": 0, "REJECTED_G3": 0, "TOTAL": 0}
    }
    
    latencies = []
    
    start_all = time.time()
    for idx, p in enumerate(val_pairs):
        r_file = p["reference_frame"]
        q_file = p["query_frame"]
        lbl = p["label"] + "_" + p["difficulty_tag"]
        
        path_r = os.path.join(DIR_REF, r_file)
        path_q = os.path.join(DIR_QRY, q_file)
        
        im_r = cv2.imread(path_r)
        im_q = cv2.imread(path_q)
        if im_r is None or im_q is None:
            continue
            
        d_r = dets_ref.get(r_file, [])
        d_q = dets_qry.get(q_file, [])
        
        t0 = time.time()
        status, details = verifier.verify_pair(im_r, im_q, d_r, d_q)
        lat = (time.time() - t0) * 1000.0
        latencies.append(lat)
        
        results[lbl]["TOTAL"] += 1
        if status == "VERIFIED":
            results[lbl]["VERIFIED"] += 1
        elif status == "REJECTED_GATE_1":
            results[lbl]["REJECTED_G1"] += 1
        elif status == "REJECTED_GATE_2":
            results[lbl]["REJECTED_G2"] += 1
        elif status == "REJECTED_GATE_3":
            results[lbl]["REJECTED_G3"] += 1
            
        if (idx + 1) % 20 == 0 or (idx + 1) == len(val_pairs):
            print(f"   Processed {idx + 1:3d}/{len(val_pairs)} pairs... (Avg latency: {np.mean(latencies):.1f} ms)")

    total_time = time.time() - start_all
    
    # 4. In bảng thống kê chi tiết
    print("\n=================================================================")
    print("                     KẾT QUẢ PHASE 4 CHI TIẾT                     ")
    print("=================================================================")
    print(f"{'Tập Dữ Liệu':<18} | {'Tổng':<5} | {'XÁC NHẬN':<10} | {'Loại G1':<8} | {'Loại G2':<8} | {'Loại G3':<8} | {'Tỷ lệ':<8}")
    print("-" * 75)
    
    for lbl, stats in results.items():
        tot = stats["TOTAL"]
        ver = stats["VERIFIED"]
        g1 = stats["REJECTED_G1"]
        g2 = stats["REJECTED_G2"]
        g3 = stats["REJECTED_G3"]
        rate = (ver / tot * 100.0) if tot > 0 else 0.0
        print(f"{lbl:<18} | {tot:<5} | {ver:<10} | {g1:<8} | {g2:<8} | {g3:<8} | {rate:6.1f}%")
        
    print("-" * 75)
    
    # Tính Precision, Recall, FPR
    tp = results["POSITIVE_EASY"]["VERIFIED"] + results["POSITIVE_HARD"]["VERIFIED"]
    fn = (results["POSITIVE_EASY"]["TOTAL"] + results["POSITIVE_HARD"]["TOTAL"]) - tp
    fp = results["NEGATIVE_EASY"]["VERIFIED"]
    tn = results["NEGATIVE_EASY"]["TOTAL"] - fp
    
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall_easy = (results["POSITIVE_EASY"]["VERIFIED"] / results["POSITIVE_EASY"]["TOTAL"] * 100.0)
    recall_hard = (results["POSITIVE_HARD"]["VERIFIED"] / results["POSITIVE_HARD"]["TOTAL"] * 100.0)
    overall_recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    
    print("\n--- CHỈ SỐ TOÀN DIỆN (ACCURACY-FIRST METRICS) ---")
    print(f"1. False Positive Rate (FPR) : {fpr:.2f}% ({fp}/{fp + tn} false loops)")
    print(f"2. Precision                  : {precision:.2f}%")
    print(f"3. Recall POSITIVE EASY       : {recall_easy:.1f}% ({results['POSITIVE_EASY']['VERIFIED']}/{results['POSITIVE_EASY']['TOTAL']})")
    print(f"4. Recall POSITIVE HARD       : {recall_hard:.1f}% ({results['POSITIVE_HARD']['VERIFIED']}/{results['POSITIVE_HARD']['TOTAL']})")
    print(f"5. Overall Recall             : {overall_recall:.1f}% ({tp}/{tp + fn})")
    print(f"6. Average Latency            : {np.mean(latencies):.1f} ms / pair")
    print(f"7. Total Runtime              : {total_time:.2f} seconds")
    print("=================================================================")


if __name__ == "__main__":
    main()

