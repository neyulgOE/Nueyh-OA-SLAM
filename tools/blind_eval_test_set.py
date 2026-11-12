#!/usr/bin/env python3
"""
Blind Evaluation on Locked Test Set (88 Pairs)
Dự án: Long-Term Visual Place Recognition for OA-SLAM
Nguyên tắc: Đánh giá mù tuyệt đối (Zero tuning trên tập Test) để đảm bảo tính trung thực thống kê.
"""

import os
import cv2
import numpy as np
import json
import csv
import time
import torch
from tools.cascading_verification import (
    CascadingVerifier,
    load_detections_map,
    DIR_REF,
    DIR_QRY,
    JSON_REF,
    JSON_QRY
)

TEST_CSV = "/home/huyen1905/OA-SLAM/Data/ground_truth_d455/test_pairs.csv"
OUTPUT_REPORT = "/home/huyen1905/OA-SLAM/Data/ground_truth_d455/blind_test_report.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("=================================================================")
    print("   BLIND EVALUATION TRÊN LOCKED TEST SET (88 PAIRS)              ")
    print("   Nguyên tắc: Giữ nguyên 100% siêu tham số từ tập Validation   ")
    print("=================================================================")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Tải JSON Detections
    print("\n1. Đang nạp JSON Detections...")
    dets_ref = load_detections_map(JSON_REF)
    dets_qry = load_detections_map(JSON_QRY)

    # 2. Khởi tạo Verifier với thông số đã khóa
    print("\n2. Khởi tạo Cascading Verifier (Đã khóa tham số)...")
    verifier = CascadingVerifier()
    print(f"   -> Min Inliers        : {verifier.min_inliers}")
    print(f"   -> Min Inlier Ratio   : {verifier.min_inlier_ratio * 100:.0f}%")
    print(f"   -> Max Epipolar Error : {verifier.max_epipolar_err} px")
    print(f"   -> Min Cheirality     : {verifier.min_cheirality_ratio * 100:.0f}%")
    print(f"   -> Min Grid Occupancy : {verifier.min_grid_occupancy}/9")
    print(f"   -> Min Hull Area Ratio: {verifier.min_hull_area_ratio * 100:.0f}%")

    # 3. Đọc Test Pairs (88 pairs)
    with open(TEST_CSV, "r", encoding="utf-8") as f:
        test_pairs = list(csv.DictReader(f))

    print(f"\n3. Bắt đầu đánh giá mù trên {len(test_pairs)} Test Pairs:")

    results = {
        "POSITIVE_EASY": {"VERIFIED": 0, "REJECTED_G1": 0, "REJECTED_G2": 0, "REJECTED_G3": 0, "TOTAL": 0},
        "POSITIVE_HARD": {"VERIFIED": 0, "REJECTED_G1": 0, "REJECTED_G2": 0, "REJECTED_G3": 0, "TOTAL": 0},
        "NEGATIVE_EASY": {"VERIFIED": 0, "REJECTED_G1": 0, "REJECTED_G2": 0, "REJECTED_G3": 0, "TOTAL": 0}
    }

    detailed_log = []
    latencies = []
    start_all = time.time()

    for idx, p in enumerate(test_pairs):
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

        detailed_log.append({
            "pair_id": p["pair_id"],
            "ref_frame": r_file,
            "qry_frame": q_file,
            "ground_truth": lbl,
            "status": status,
            "latency_ms": lat,
            "details": details
        })

        if (idx + 1) % 20 == 0 or (idx + 1) == len(test_pairs):
            print(f"   Processed {idx + 1:2d}/{len(test_pairs)} pairs... (Avg latency: {np.mean(latencies):.1f} ms)")

    total_time = time.time() - start_all

    # 4. In bảng kết quả
    print("\n=================================================================")
    print("              KẾT QUẢ BLIND EVALUATION TRÊN TEST SET              ")
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

    # Thống kê chỉ số khoa học
    tp = results["POSITIVE_EASY"]["VERIFIED"] + results["POSITIVE_HARD"]["VERIFIED"]
    fn = (results["POSITIVE_EASY"]["TOTAL"] + results["POSITIVE_HARD"]["TOTAL"]) - tp
    fp = results["NEGATIVE_EASY"]["VERIFIED"]
    tn = results["NEGATIVE_EASY"]["TOTAL"] - fp

    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall_easy = (results["POSITIVE_EASY"]["VERIFIED"] / results["POSITIVE_EASY"]["TOTAL"] * 100.0)
    recall_hard = (results["POSITIVE_HARD"]["VERIFIED"] / results["POSITIVE_HARD"]["TOTAL"] * 100.0)
    overall_recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0

    print("\n--- CHỈ SỐ KHOA HỌC TRÊN TẬP TEST ĐỘC LẬP (UNSEEN DATA) ---")
    print(f"1. False Positive Rate (FPR) : {fpr:.2f}% ({fp}/{fp + tn} false loops)")
    print(f"2. Precision                  : {precision:.2f}% ({tp}/{tp + fp})")
    print(f"3. Recall POSITIVE EASY       : {recall_easy:.1f}% ({results['POSITIVE_EASY']['VERIFIED']}/{results['POSITIVE_EASY']['TOTAL']})")
    print(f"4. Recall POSITIVE HARD       : {recall_hard:.1f}% ({results['POSITIVE_HARD']['VERIFIED']}/{results['POSITIVE_HARD']['TOTAL']})")
    print(f"5. Overall Recall             : {overall_recall:.1f}% ({tp}/{tp + fn})")
    print(f"6. Average Latency            : {np.mean(latencies):.1f} ms / pair")
    print(f"7. Total Runtime              : {total_time:.2f} seconds")
    print("=================================================================")

    # Lưu log chi tiết
    report_data = {
        "summary": {
            "test_pairs_total": len(test_pairs),
            "false_positive_rate": fpr,
            "precision": precision,
            "recall_positive_easy": recall_easy,
            "recall_positive_hard": recall_hard,
            "overall_recall": overall_recall,
            "avg_latency_ms": float(np.mean(latencies)),
            "total_time_seconds": total_time
        },
        "breakdown": results,
        "detailed_log": detailed_log
    }
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\nĐã xuất báo cáo kiểm định chi tiết tại: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()

