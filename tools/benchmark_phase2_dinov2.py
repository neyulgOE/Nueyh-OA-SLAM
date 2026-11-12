#!/usr/bin/env python3
"""
Phase 2: Global Place Recognition Benchmark - Model 2 (DINOv2 / AnyLoc Backbone)
Dự án: Long-term Visual Place Recognition for OA-SLAM

Quy trình:
1. Tải DINOv2 (dinov2_vits14) chạy trên NVIDIA GeForce RTX 4060 GPU.
2. Trích xuất Global Descriptor (384-d vector chuẩn hóa L2) cho Reference Database (272 KeyFrames).
3. Đánh giá Top-K Retrieval bằng Cosine Similarity trên tập Validation:
   - POSITIVE - EASY (Lượt đi).
   - POSITIVE - HARD (Lượt về đóng vòng sau 4000 frame).
   - NEGATIVE (Các frame ở phân đoạn đi xa).
4. So sánh trực tiếp với Baseline ORB!
"""

import torch
import torchvision.transforms as T
import cv2
import numpy as np
import os
import csv
import time

DIR_REF = "/home/huyen1905/OA-SLAM/Data/dataset/images"
DIR_QRY = "/home/huyen1905/OA-SLAM/Data/dataset2/images"
VAL_CSV = "/home/huyen1905/OA-SLAM/Data/ground_truth_d455/validation_pairs.csv"

KF_STEP = 5
TOLERANCE_FRAMES = 15

# Chuẩn bị transform cho DINOv2 (chuẩn ImageNet)
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_dinov2_descriptor(img_bgr, model):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor) # [1, 384]
        # L2 Normalize
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy()

def main():
    print("=================================================================")
    print("   PHASE 2 BENCHMARK: MODEL 2 — DINOv2 (AnyLoc Backbone)         ")
    print("=================================================================")
    print(f"Thiết bị tính toán: {device} ({torch.cuda.get_device_name(0)})")
    
    # 1. Tải model DINOv2 đã cache
    t0 = time.time()
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device).eval()
    print(f"Model DINOv2 loaded in {time.time()-t0:.2f}s")

    # 2. Xây dựng Reference Database
    ref_files = sorted(os.listdir(DIR_REF))
    db_indices = []
    db_descriptors = []
    db_fnames = []
    
    print(f"\n1. Đang trích xuất DINOv2 Descriptors cho Reference Database ({len(ref_files)} frames, step {KF_STEP})...")
    t0 = time.time()
    for idx in range(0, len(ref_files), KF_STEP):
        fname = ref_files[idx]
        img = cv2.imread(os.path.join(DIR_REF, fname))
        if img is not None:
            des = extract_dinov2_descriptor(img, model)
            db_indices.append(idx)
            db_descriptors.append(des)
            db_fnames.append(fname)
            
    db_descriptors = np.array(db_descriptors) # [N_kf, 384]
    print(f"   -> Đã index {len(db_descriptors)} KeyFrames ({time.time()-t0:.2f}s, TB: {(time.time()-t0)/len(db_descriptors)*1000:.1f}ms/frame)")
    print(f"   -> Kích thước Descriptor DB: {db_descriptors.nbytes / 1024:.1f} KB (Siêu nhẹ!)")

    # 3. Đọc Validation Pairs
    with open(VAL_CSV, "r", encoding="utf-8") as f:
        val_pairs = list(csv.DictReader(f))

    pos_easy = [r for r in val_pairs if r["label"] == "POSITIVE" and r["difficulty_tag"] == "EASY"]
    pos_hard = [r for r in val_pairs if r["label"] == "POSITIVE" and r["difficulty_tag"] == "HARD"]
    neg_pairs = [r for r in val_pairs if r["label"] == "NEGATIVE"]

    print(f"\n2. Tập Validation: {len(val_pairs)} cặp")
    print(f"   - Positive EASY (Lượt đi)           : {len(pos_easy)} queries")
    print(f"   - Positive HARD (Loop sau 4000 frame): {len(pos_hard)} queries")
    print(f"   - Negative (Đi xa)                  : {len(neg_pairs)} queries")

    # Hàm truy vấn Top-K bằng Cosine Similarity
    def query_top_k(q_des, top_k=20):
        # q_des: [384], db_descriptors: [N_kf, 384] -> dot product = cosine similarity
        sims = np.dot(db_descriptors, q_des)
        # Sắp xếp giảm dần
        ranked_indices = np.argsort(sims)[::-1][:top_k]
        return [(sims[i], db_indices[i], db_fnames[i]) for i in ranked_indices]

    # 4. Đánh giá trên Positive Pairs
    def evaluate_positive(pairs, group_name):
        print(f"\n--- Đang đánh giá {group_name} ({len(pairs)} queries) ---")
        recalls = {1: 0, 5: 0, 10: 0, 20: 0}
        latencies = []

        for p in pairs:
            q_file = p["query_frame"]
            target_r_file = p["reference_frame"]
            target_r_idx = int(target_r_file.split(".")[0])

            img_q = cv2.imread(os.path.join(DIR_QRY, q_file))
            if img_q is None:
                continue

            t_start = time.time()
            q_des = extract_dinov2_descriptor(img_q, model)
            top_results = query_top_k(q_des, top_k=20)
            latencies.append((time.time() - t_start) * 1000)

            retrieved_indices = [r[1] for r in top_results]
            for k in [1, 5, 10, 20]:
                subset = retrieved_indices[:k]
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
        print(f"  Độ trễ TB (Feature + Search): {avg_lat:.2f} ms/query")
        return {"r1": r1, "r5": r5, "r10": r10, "r20": r20, "lat": avg_lat}

    res_easy = evaluate_positive(pos_easy, "POSITIVE - EASY (Lượt đi)")
    res_hard = evaluate_positive(pos_hard, "POSITIVE - HARD (Loop sau 4000 frames)")

    # 5. Đánh giá trên Negative Pairs
    print(f"\n--- Đang đánh giá NEGATIVE ({len(neg_pairs)} queries) ---")
    neg_top_sims = []
    for p in neg_pairs:
        q_file = p["query_frame"]
        img_q = cv2.imread(os.path.join(DIR_QRY, q_file))
        if img_q is not None:
            q_des = extract_dinov2_descriptor(img_q, model)
            top_res = query_top_k(q_des, top_k=1)
            neg_top_sims.append(top_res[0][0])

    avg_neg_sim = np.mean(neg_top_sims) if neg_top_sims else 0
    max_neg_sim = np.max(neg_top_sims) if neg_top_sims else 0
    print(f"  Cosine Similarity cao nhất trên Negative TB : {avg_neg_sim:.3f}")
    print(f"  Cosine Similarity cao nhất từng gặp        : {max_neg_sim:.3f}")

    # 6. Bảng so sánh tổng hợp Phase 2: ORB vs DINOv2
    print("\n================================================================================")
    print("           BẢNG SO SÁNH TỔNG HỢP PHASE 2: ORB BASELINE vs DINOv2 (AnyLoc)        ")
    print("================================================================================")
    print(f"| Phương pháp | Kịch bản | Recall@1 | Recall@5 | Recall@10 | Độ trễ / query | Kích thước DB |")
    print(f"|:---|:---|:---:|:---:|:---:|:---:|:---:|")
    print(f"| **ORB Baseline** | Positive EASY | 100.0% | 100.0% | 100.0% | 375.7 ms | ~2.5 MB |")
    print(f"| **ORB Baseline** | Positive HARD |  81.8% | 100.0% | 100.0% | 366.2 ms | ~2.5 MB |")
    print(f"| **DINOv2 (AnyLoc)** | Positive EASY | **{res_easy['r1']:.1f}%** | **{res_easy['r5']:.1f}%** | **{res_easy['r10']:.1f}%** | **{res_easy['lat']:.1f} ms** | **408 KB** |")
    print(f"| **DINOv2 (AnyLoc)** | Positive HARD | **{res_hard['r1']:.1f}%** | **{res_hard['r5']:.1f}%** | **{res_hard['r10']:.1f}%** | **{res_hard['lat']:.1f} ms** | **408 KB** |")
    print("================================================================================")

if __name__ == "__main__":
    main()

